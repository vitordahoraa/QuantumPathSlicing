"""
qaoa_calibracao.py
===========================================================================
Onde os ângulos (gamma, beta) do QAOA são obtidos. É a ÚNICA parte do
projeto com um otimizador clássico (COBYLA), e por isso vive isolada do
módulo quântico, para que a fronteira seja fácil de auditar:

  - ProvedorCalibracao : calibra os ângulos de cada fatia com um laço
                         variacional (circuito -> valor esperado -> COBYLA)
                         e os SALVA em angulos.json;
  - ProvedorLookup    : NÃO otimiza nada; só LÊ do angulos.json os ângulos
                         já calibrados e levanta erro se algum faltar.

O quantum_path_slicing.py recebe qualquer um dos dois como 'provedor_angulos'
e só monta o circuito com os ângulos que chegam.

Cada entrada do arquivo é identificada por um hash dos coeficientes do QUBO
(mais o número de camadas): os ângulos só valem para o problema para o qual
foram calibrados. Se a fatia mudar — outra instância, outro K, outra
penalidade, ou um t_origem diferente porque a fatia anterior decodificou
outra ordem —, o hash muda e o modo lookup recusa em vez de usar ângulos
que não casam.
"""

import hashlib
import json
import os
import time as time_module
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from quantum_path_slicing import (
    energias_qubo,
    montar_circuito_qaoa,
    normalizar_hamiltoniano,
)

ARQUIVO_PADRAO = "angulos.json"
VERSAO_ARQUIVO = 1


class AngulosNaoEncontradosError(RuntimeError):
    """O modo lookup não achou ângulos salvos para a fatia."""


# ---------------------------------------------------------------------------
# Identificação do problema e armazenamento
# ---------------------------------------------------------------------------

def chave_qubo(Q, n_vars, camadas):
    """Hash estável do QUBO (coeficientes arredondados) + camadas."""
    itens = sorted(
        (int(a), int(b), f"{round(float(c), 6) + 0.0:.6f}")
        for (a, b), c in Q.items()
    )
    texto = json.dumps([int(n_vars), int(camadas), itens], separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def carregar_arquivo(caminho):
    """Lê o arquivo de ângulos. Devolve {} se ele não existir."""
    caminho = Path(caminho)
    if not caminho.is_file():
        return {}
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return dados.get("angulos", {})


def salvar_entrada(caminho, chave, entrada):
    """Acrescenta/atualiza UMA entrada, preservando as demais (o arquivo
    acumula várias instâncias) e gravando de forma atômica, para que uma
    execução interrompida não corrompa o arquivo."""
    caminho = Path(caminho)
    angulos = carregar_arquivo(caminho)
    angulos[chave] = entrada
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"versao": VERSAO_ARQUIVO, "angulos": angulos},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, caminho)


# ---------------------------------------------------------------------------
# Calibração (laço variacional com COBYLA)
# ---------------------------------------------------------------------------

def calibrar_angulos(dispositivo, n_vars, h, Jm, offset, camadas=1,
                     amostras=500, maxiter=25, seed=42, exato=False):
    """Loop variacional híbrido: dispara o circuito parametrizado no
    'dispositivo' e otimiza (gamma, beta) classicamente via COBYLA,
    replicando o Quadro 6 do TCC. A função de custo é o valor esperado da
    energia do QUBO sobre a distribuição medida.

    'exato=True' usa a distribuição exata do vetor de estado (shots=0) para
    avaliar o valor esperado — sem ruído de amostragem, o que ajuda muito o
    COBYLA. Só é possível em simulador.

    Devolve (gamma, beta, tempo), com gamma e beta listas de 'camadas'
    ângulos."""
    shots = 0 if exato else amostras
    h_circ, Jm_circ = normalizar_hamiltoniano(h, Jm)

    def custo_esperado(params):
        circ = montar_circuito_qaoa(
            n_vars, h_circ, Jm_circ, params[:camadas], params[camadas:], camadas
        )
        probs = dispositivo.probabilidades(circ, shots)
        ativos = np.flatnonzero(probs)
        if ativos.size == 0:
            return 0.0
        return float(
            np.dot(probs[ativos], energias_qubo(ativos, n_vars, h, Jm, offset))
        )

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=2 * camadas)

    t0 = time_module.time()
    res = minimize(custo_esperado, x0, method="COBYLA", options={"maxiter": maxiter})
    tempo = time_module.time() - t0

    return (
        [float(g) for g in res.x[:camadas]],
        [float(b) for b in res.x[camadas:]],
        tempo,
    )


# ---------------------------------------------------------------------------
# Provedores de ângulos (o que o pipeline quântico recebe)
# ---------------------------------------------------------------------------

class ProvedorCalibracao:
    """Calibra os ângulos de cada fatia com COBYLA e os salva em disco."""

    modo = "calibrado"

    def __init__(self, dispositivo, caminho=ARQUIVO_PADRAO, camadas=1,
                 amostras=500, maxiter=25, seed=42, exato=False):
        self.dispositivo = dispositivo
        self.caminho = caminho
        self.camadas = camadas
        self.amostras = amostras
        self.maxiter = maxiter
        self.seed = seed
        self.exato = exato

    def __call__(self, Q, n_vars, h, Jm, offset, contexto):
        gamma, beta, tempo = calibrar_angulos(
            self.dispositivo, n_vars, h, Jm, offset, camadas=self.camadas,
            amostras=self.amostras, maxiter=self.maxiter, seed=self.seed,
            exato=self.exato,
        )
        salvar_entrada(
            self.caminho,
            chave_qubo(Q, n_vars, self.camadas),
            dict(
                gamma=gamma, beta=beta, camadas=self.camadas, n_vars=n_vars,
                fatia=[int(x) for x in contexto["fatia"]],
                t_origem=float(contexto["t_origem"]),
            ),
        )
        return gamma, beta, tempo


class ProvedorLookup:
    """Lê os ângulos já calibrados. Não otimiza nada e não tem plano B: se a
    fatia não estiver no arquivo, levanta AngulosNaoEncontradosError."""

    modo = "lookup"

    def __init__(self, caminho=ARQUIVO_PADRAO, camadas=1):
        self.caminho = caminho
        self.camadas = camadas
        self.angulos = carregar_arquivo(caminho)
        if not self.angulos:
            raise AngulosNaoEncontradosError(
                f"Nenhum ângulo salvo em {caminho!r}. Rode primeiro sem "
                f"--lookup para calibrar e gerar o arquivo."
            )

    def __call__(self, Q, n_vars, h, Jm, offset, contexto):
        chave = chave_qubo(Q, n_vars, self.camadas)
        entrada = self.angulos.get(chave)
        if entrada is None:
            raise AngulosNaoEncontradosError(
                f"Sem ângulos salvos para a fatia {list(contexto['fatia'])} "
                f"(t_origem={contexto['t_origem']:.1f}, {n_vars} variáveis, "
                f"{self.camadas} camada(s), chave {chave}) em "
                f"{str(self.caminho)!r}. O QUBO desta fatia difere do que foi "
                f"calibrado (outra instância/parâmetro, ou a fatia anterior "
                f"decodificou outra ordem). Rode sem --lookup para recalibrar."
            )
        return entrada["gamma"], entrada["beta"], 0.0
