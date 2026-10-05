"""
Avaliação de desempenho de uma abordagem híbrida quântica de busca local
aplicada ao TSPTW (Traveling Salesman Problem with Time Windows)
------------------------------------------------------------------------
Script básico de demonstração, seguindo a arquitetura descrita no TCC:

    Carregamento das instâncias -> Path-Slicing -> Tradutor QUBO
    -> Execução Quântica (QAOA no Amazon Braket LocalSimulator)
    -> Pós-processamento (busca local 2-opt) -> Análise

Observações sobre esta versão de demonstração:
- Usa a instância Solomon R101 (adaptada para veículo único / capacidade
  infinita, como no TCC), mas com um recorte pequeno de nós para que o
  circuito QAOA (variáveis binárias x_{i,k}, K^2 qubits por fatia) rode em
  tempo razoável em um simulador de vetor de estado local.
- O recorte é feito SEM nenhum pré-filtro de viabilidade: pegam-se
  simplesmente os n_clientes primeiros nós (por ordem de ID) após o
  depósito. Se o subconjunto resultante não admitir nenhuma ordem de
  visita viável, isso é tratado no final, na etapa de simulação do
  percurso.
- A formulação QUBO é uma versão simplificada da apresentada no Capítulo 4
  do TCC: mantém os componentes de Custo (H_Custo) e Fluxo/Unicidade
  (H_Fluxo) de forma fiel, e usa uma penalidade de precedência temporal
  simplificada (sem variáveis de folga/ancilla) para H_Tempo, penalizando
  transições i->j que necessariamente violam a janela de tempo de j.
- QAOA é implementado "na mão" (Hamiltoniano de custo diagonal em Z,
  camada de mixing com Rx) usando braket.circuits.Circuit e executado no
  braket.devices.LocalSimulator ("braket_sv"), com otimização clássica dos
  ângulos via scipy.optimize (COBYLA), replicando o loop variacional do
  Quadro 6 do TCC.
- Ao final, a rota é EXECUTADA nó a nó com o relógio real. Mesmo que a
  solução seja inviável, o veículo tenta percorrer o máximo possível: na
  primeira janela estourada o percurso é interrompido e são reportados
  (i) o caminho efetivamente percorrido até o último nó visitado e
  (ii) a lista de nós que não foi possível visitar.

Requisitos:
    pip install amazon-braket-sdk numpy scipy networkx
"""

import os
import math
import itertools
import time as time_module

import numpy as np
import networkx as nx
from scipy.optimize import minimize

from braket.circuits import Circuit
from braket.devices import LocalSimulator


# ---------------------------------------------------------------------------
# 4.3.1 Carregamento das instâncias
# ---------------------------------------------------------------------------

# Formato de benchmark TSPTW usado aqui (Dumas / Gendreau / Solomon-derivados,
# ex.: "rc_201.2.txt"), baseado em MATRIZ DE TEMPOS DE VIAGEM em vez de
# coordenadas cartesianas:
#
#     linha 1        : n (número de nós, incluindo o depósito 0)
#     n linhas       : matriz n x n de tempos de viagem M[i][j]
#     n linhas       : janela de tempo de cada nó, "a_i  b_i"
#
# Convenção destes arquivos: o tempo de serviço do nó i já está EMBUTIDO na
# linha i da matriz, isto é
#
#     M[i][j] = d(i, j) + s_i        e        M[i][i] = s_i
#
# (é por isso que a diagonal vale 10 para os clientes e 0 para o depósito, e
# que M[0][1] = 33.541 enquanto M[1][0] = 43.541 — a diferença é exatamente
# s_1 = 10). Para que o restante do pipeline continue somando o serviço
# explicitamente (chegada = t + servico[i] + C[i][j]), o parser SEPARA as duas
# parcelas: extrai s_i da diagonal e devolve C[i][j] = M[i][j] - s_i, que é a
# matriz de distâncias/tempos puros (simétrica, como esperado).

# Diretório (relativo a ESTE arquivo .py) onde ficam os .txt de benchmark, e
# caminho relativo da instância que será resolvida por padrão. Basta trocar a
# string abaixo para rodar outra instância da pasta.
PASTA_BENCHMARK = "benchmark"
INSTANCIA_PADRAO = "benchmark/rc_204.3.txt"

# raiz usada para resolver caminhos relativos: a pasta onde este script está
# salvo (e não o diretório de trabalho atual), para que o script funcione
# mesmo sendo chamado de outro lugar, ex.: `python pasta/main4.py`
RAIZ_SCRIPT = os.path.dirname(os.path.abspath(__file__))


def resolver_caminho_instancia(caminho):
    """Converte um caminho relativo (ex.: "benchmark/rc_201.2.txt" ou apenas
    "rc_201.2.txt") no caminho absoluto do arquivo, procurando, nesta ordem:

        1. o caminho exatamente como foi dado (absoluto ou relativo ao cwd);
        2. o caminho relativo à pasta deste script;
        3. o caminho dentro de <pasta do script>/PASTA_BENCHMARK.

    Levanta FileNotFoundError listando os locais tentados se nada for achado."""
    caminho = os.path.normpath(str(caminho))

    if os.path.isabs(caminho):
        candidatos = [caminho]
    else:
        candidatos = [
            os.path.abspath(caminho),
            os.path.join(RAIZ_SCRIPT, caminho),
            os.path.join(RAIZ_SCRIPT, PASTA_BENCHMARK, os.path.basename(caminho)),
        ]

    for candidato in candidatos:
        if os.path.isfile(candidato):
            return candidato

    tentativas = "\n".join(f"      - {c}" for c in dict.fromkeys(candidatos))
    raise FileNotFoundError(
        f"Instância não encontrada: {caminho!r}. Locais procurados:\n{tentativas}"
    )


def _parece_caminho(texto):
    """Distingue um caminho de arquivo do conteúdo bruto de uma instância."""
    return "\n" not in texto and len(texto.strip()) < 4096 and texto.strip() != ""


def _ler_texto_instancia(fonte):
    """Aceita um caminho de arquivo (absoluto ou relativo) OU o próprio
    conteúdo textual da instância."""
    if isinstance(fonte, (bytes, bytearray)):
        return fonte.decode("utf-8")

    texto = str(fonte)
    if _parece_caminho(texto):
        caminho = resolver_caminho_instancia(texto.strip())
        with open(caminho, "r", encoding="utf-8") as arq:
            return arq.read()
    return texto


def parse_instancia_tsptw(fonte):
    """Faz a ingestão e o parsing do arquivo de benchmark no formato de
    matriz, com validação estrutural (contagem de tokens, diagonal, janelas
    a_i <= b_i).

    Retorna (M, C, A, B, servico, n), onde M é a matriz original do arquivo,
    C = M - s_i (tempos de viagem puros) e servico = diag(M)."""
    tokens = _ler_texto_instancia(fonte).split()
    if not tokens:
        raise ValueError("Instância vazia.")

    n = int(tokens[0])
    esperado = 1 + n * n + 2 * n
    if len(tokens) < esperado:
        raise ValueError(
            f"Instância malformada: n={n} exige {esperado} números "
            f"(1 + {n}x{n} + 2x{n}), mas foram lidos {len(tokens)}."
        )

    valores = np.array([float(t) for t in tokens[1:esperado]], dtype=float)
    M = valores[: n * n].reshape(n, n)
    janelas = valores[n * n :].reshape(n, 2)

    A = janelas[:, 0].copy()
    B = janelas[:, 1].copy()

    # tempo de serviço embutido na diagonal; C = viagem pura
    servico = np.diag(M).copy()
    C = M - servico[:, None]
    np.fill_diagonal(C, 0.0)

    if not np.all(A <= B):
        raise ValueError("Instância inválida: existe a_i > b_i.")
    if np.any(C < -1e-6):
        raise ValueError("Instância inválida: tempos de viagem negativos.")

    return M, C, A, B, servico, n


def _montar_instancia(C, A, B, servico, indices_originais):
    """Monta o dicionário usado pelo restante do pipeline a partir das
    matrizes/vetores já recortados, reindexando os nós em 0..n-1.

    O campo 'id_original' preserva o índice do nó no arquivo de benchmark,
    útil para reportar a solução na numeração original."""
    n = len(A)
    nos = [
        dict(
            id=novo_id,
            id_original=int(indices_originais[novo_id]),
            a=float(A[novo_id]),
            b=float(B[novo_id]),
            servico=float(servico[novo_id]),
            # esta família de benchmarks não fornece coordenadas (x, y):
            # a geometria vem pronta na matriz de tempos de viagem.
            x=None,
            y=None,
        )
        for novo_id in range(n)
    ]
    assert np.all(A <= B), "Instância inválida: existe ai > bi."
    return dict(
        nos=nos,
        C=np.asarray(C, dtype=float),
        A=np.asarray(A, dtype=float),
        B=np.asarray(B, dtype=float),
        servico=np.asarray(servico, dtype=float),
        n=n,
    )


def carregar_instancia(fonte=INSTANCIA_PADRAO, n_clientes=None):
    """Realiza a ingestão, parsing e validação estrutural dos dados do
    benchmark TSPTW no formato de matriz (ver cabeçalho desta seção),
    isolando a matriz de tempos de viagem C, os tempos de serviço e as
    janelas de tempo [a_i, b_i], e produzindo os vetores A (mínimos) e
    B (máximos).

    'fonte' pode ser o caminho do arquivo — absoluto ou relativo, ex.:
    "benchmark/rc_201.2.txt" — ou o próprio conteúdo do arquivo como string.
    Caminhos relativos são resolvidos por resolver_caminho_instancia().

    Se n_clientes for informado, o recorte é feito de forma direta: o
    depósito mais os n_clientes PRIMEIROS clientes da instância (ordem de
    leitura / ID), extraindo a submatriz correspondente. Não há nenhum
    pré-filtro de viabilidade — o subconjunto pode perfeitamente não admitir
    nenhuma ordem de visita viável, e isso será detectado e reportado na
    simulação do percurso, ao final do pipeline."""
    _, C, A, B, servico, n = parse_instancia_tsptw(fonte)

    indices = list(range(n))
    if n_clientes is not None:
        disponiveis = n - 1
        if n_clientes > disponiveis:
            print(
                f"    [aviso] Foram solicitados {n_clientes} clientes, mas a "
                f"instância só possui {disponiveis}; usando todos."
            )
            n_clientes = disponiveis
        indices = [0] + list(range(1, 1 + n_clientes))

    idx = np.array(indices, dtype=int)
    C_rec = C[np.ix_(idx, idx)]
    return _montar_instancia(C_rec, A[idx], B[idx], servico[idx], idx)


def gerar_rota_heuristica_chegada_mais_proxima(instancia):
    """Heurística construtiva gulosa "vizinho mais próximo com viabilidade
    de janela" (nearest neighbor time-feasible), conforme Seção 4.4.1 do
    TCC. A cada passo, entre os clientes ainda não visitados, escolhe-se o
    que resulta no MENOR tempo de chegada a partir do nó atual, priorizando
    os que ainda são fisicamente viáveis (chegada <= bi); se nenhum for
    viável, escolhe-se o de menor atraso.

    Note que a rota SEMPRE inclui todos os clientes, mesmo que alguns só
    possam ser alcançados com atraso: a detecção de inviabilidade fica a
    cargo da simulação final do percurso."""
    nos = instancia["nos"]
    A, servico, C = instancia["A"], instancia["servico"], instancia["C"]

    depósito = nos[0]["id"]
    nao_visitados = {no["id"] for no in nos[1:]}
    rota = [depósito]
    t_atual = A[depósito]
    atual = depósito

    while nao_visitados:
        candidatos = []
        for c in nao_visitados:
            chegada = max(t_atual + servico[atual] + C[atual, c], A[c])
            atraso = max(0.0, chegada - instancia["B"][c])
            candidatos.append((atraso, chegada, c))
        # prioriza: sem atraso primeiro, depois menor tempo de chegada
        atraso, chegada, escolhido = min(candidatos, key=lambda x: (x[0], x[1]))
        rota.append(escolhido)
        nao_visitados.remove(escolhido)
        t_atual, atual = chegada, escolhido

    return rota


# ---------------------------------------------------------------------------
# 4.3.2 Path-Slicing
# ---------------------------------------------------------------------------

def path_slicing(rota, K):
    """Decompõe a rota global em subproblemas de tamanho fixo K, com
    sobreposição de 1 nó entre fatias consecutivas (o último nó da fatia M
    é fixado como origem da fatia M+1), conforme Seção 4.3.2/4.4.1."""
    fatias = []
    i = 0
    while i < len(rota) - 1:
        fatia = rota[i : i + K]
        if len(fatia) < 2:
            break
        fatias.append(fatia)
        i += K - 1  # sobreposição de 1 nó
    return fatias


# ---------------------------------------------------------------------------
# 4.3.3 / 4.5 Tradutor QUBO
# ---------------------------------------------------------------------------

def construir_qubo(sub_rota, instancia, alpha=8.0, beta=8.0):
    """Constrói o modelo QUBO H_Total = H_Custo + alpha*H_Fluxo + beta*H_Tempo
    para a fatia (sub-rota), usando variáveis binárias x_{i,k} (cliente i
    ocupa a posição k), conforme Seção 4.5 do TCC.

    O primeiro nó da fatia é fixado como origem (posição 0), reduzindo o
    número de variáveis a otimizar para (K-1) x (K-1).

    Retorna um dicionário QUBO no formato {(var_i, var_j): coeficiente},
    compatível com a representação esparsa usada pela biblioteca dimod, e o
    mapa de índices variável <-> (cliente, posição).
    """
    C = instancia["C"]
    A = instancia["A"]
    B = instancia["B"]
    servico = instancia["servico"]

    origem = sub_rota[0]
    livres = sub_rota[1:]  # clientes cuja posição será otimizada
    K = len(livres)  # número de posições livres (1..K)

    # var_id(i, k): cliente 'livres[i]' na posição k (posições 1..K)
    def var_id(i, k):
        return i * K + (k - 1)

    Q = {}

    def soma(chave, valor):
        Q[chave] = Q.get(chave, 0.0) + valor

    # --- H_Custo: distância entre clientes consecutivos --------------------
    # posição 0 é a origem fixa; para k=1, custo de origem -> livres[i]
    for i, cli in enumerate(livres):
        soma((var_id(i, 1), var_id(i, 1)), C[origem, cli])

    for k in range(1, K):
        for i, ci in enumerate(livres):
            for j, cj in enumerate(livres):
                if i == j:
                    continue
                custo = C[ci, cj]
                v1, v2 = var_id(i, k), var_id(j, k + 1)
                chave = (min(v1, v2), max(v1, v2))
                soma(chave, custo)

    # --- H_Fluxo: cada cliente exatamente 1 posição; cada posição 1 cliente
    # Penalidade (sum_k x_ik - 1)^2 por cliente
    for i in range(K):
        for k1 in range(1, K + 1):
            v1 = var_id(i, k1)
            soma((v1, v1), soma_penal_linear(alpha))
            for k2 in range(1, K + 1):
                v2 = var_id(i, k2)
                if k1 == k2:
                    continue
                a1, a2 = min(v1, v2), max(v1, v2)
                soma((a1, a2), alpha)

    # Penalidade (sum_i x_ik - 1)^2 por posição
    for k in range(1, K + 1):
        for i1 in range(K):
            v1 = var_id(i1, k)
            soma((v1, v1), soma_penal_linear(alpha))
            for i2 in range(K):
                v2 = var_id(i2, k)
                if i1 == i2:
                    continue
                a1, a2 = min(v1, v2), max(v1, v2)
                soma((a1, a2), alpha)

    # --- H_Tempo (simplificada): penaliza transições i->j entre posições
    # consecutivas que, considerando o tempo de serviço e o deslocamento,
    # tornam fisicamente impossível respeitar a janela superior b_j, isto
    # é, quando a_i + servico_i + c_ij > b_j (não há folga possível).
    # Esta é a versão simplificada de H_Tempo (Equação 24 do TCC), sem as
    # variáveis de folga (slack), adequada a um script básico de demo.
    for k in range(1, K):
        for i, ci in enumerate(livres):
            for j, cj in enumerate(livres):
                if i == j:
                    continue
                chegada_min = A[ci] + servico[ci] + C[ci, cj]
                if chegada_min > B[cj]:
                    v1, v2 = var_id(i, k), var_id(j, k + 1)
                    chave = (min(v1, v2), max(v1, v2))
                    soma(chave, beta)

    n_vars = K * K
    mapa = {var_id(i, k): (livres[i], k) for i in range(K) for k in range(1, K + 1)}
    return Q, n_vars, mapa, origem


def soma_penal_linear(alpha):
    """Termo linear da expansão de (sum x -1)^2 = sum x_i -2 sum x_i + 1
    -> coeficiente diagonal de cada x_i é -alpha (constante omitida)."""
    return -alpha


# ---------------------------------------------------------------------------
# QUBO -> Hamiltoniano de Custo (Pauli-Z) - Seção 2.4.1, Equação 8
# ---------------------------------------------------------------------------

def qubo_para_ising(Q, n_vars):
    """Converte o QUBO (variáveis x_i em {0,1}) para os coeficientes do
    Hamiltoniano de Ising equivalente (variáveis z_i em {-1,1}), usando
    x_i = (1 - z_i) / 2 (Equação 8 do TCC)."""
    h = np.zeros(n_vars)
    Jm = np.zeros((n_vars, n_vars))
    offset = 0.0

    for (a, b), coef in Q.items():
        if a == b:
            # x_a = (1 - z_a)/2  =>  coef*x_a = coef/2 - (coef/2) z_a
            offset += coef / 2
            h[a] += -coef / 2
        else:
            # x_a x_b = (1 - z_a)(1 - z_b)/4
            #         = 1/4 - z_a/4 - z_b/4 + z_a z_b/4
            offset += coef / 4
            h[a] += -coef / 4
            h[b] += -coef / 4
            Jm[a, b] += coef / 4

    return h, Jm, offset


def energia_ising(bits, h, Jm, offset):
    z = 1 - 2 * bits  # bit 0 -> z=+1 ; bit 1 -> z=-1
    e = offset + np.dot(h, z)
    idx = np.argwhere(Jm != 0)
    for a, b in idx:
        e += Jm[a, b] * z[a] * z[b]
    return e


# ---------------------------------------------------------------------------
# 4.3.4 Execução Quântica: QAOA no Amazon Braket LocalSimulator
# ---------------------------------------------------------------------------

def montar_circuito_qaoa(n_vars, h, Jm, gamma, beta_ang, camadas):
    circ = Circuit()
    for q in range(n_vars):
        circ.h(q)

    for p in range(camadas):
        g = gamma[p]
        # operador de custo: exp(-i*g*H_C)
        for q in range(n_vars):
            if h[q] != 0:
                circ.rz(q, 2 * g * h[q])
        idx = np.argwhere(Jm != 0)
        for a, b in idx:
            if a < b:
                jab = Jm[a, b]
                circ.cnot(int(a), int(b))
                circ.rz(int(b), 2 * g * jab)
                circ.cnot(int(a), int(b))
        # operador de mixing: exp(-i*b*sum X)
        bta = beta_ang[p]
        for q in range(n_vars):
            circ.rx(q, 2 * bta)

    circ.probability()
    return circ


def executar_qaoa(n_vars, h, Jm, offset, camadas=1, amostras=500, maxiter=25):
    """Loop variacional híbrido: dispara o circuito parametrizado no
    LocalSimulator do Amazon Braket e otimiza (gamma, beta) classicamente
    via COBYLA (scipy), conforme Quadro 6 do TCC."""
    device = LocalSimulator()

    def custo_esperado(params):
        gamma = params[:camadas]
        beta_ang = params[camadas:]
        circ = montar_circuito_qaoa(n_vars, h, Jm, gamma, beta_ang, camadas)
        resultado = device.run(circ, shots=amostras).result()
        probs = resultado.values[0]

        e_total = 0.0
        for bit_idx, p in enumerate(probs):
            if p == 0:
                continue
            bits = np.array(
                [int(b) for b in format(bit_idx, f"0{n_vars}b")]
            )
            e_total += p * energia_ising(bits, h, Jm, offset)
        return e_total

    rng = np.random.default_rng(42)
    x0 = rng.uniform(0, np.pi, size=2 * camadas)

    t0 = time_module.time()
    res = minimize(
        custo_esperado, x0, method="COBYLA", options={"maxiter": maxiter}
    )
    tempo_simulacao = time_module.time() - t0

    gamma_opt = res.x[:camadas]
    beta_opt = res.x[camadas:]
    circ_final = montar_circuito_qaoa(n_vars, h, Jm, gamma_opt, beta_opt, camadas)
    resultado = device.run(circ_final, shots=amostras).result()
    probs = resultado.values[0]

    melhor_idx = int(np.argmax(probs))
    bitstring_otima = format(melhor_idx, f"0{n_vars}b")

    return bitstring_otima, tempo_simulacao


# ---------------------------------------------------------------------------
# 4.3.5 Pós-processamento: decodificação + busca local 2-opt
# ---------------------------------------------------------------------------

def decodificar_bitstring(bitstring, mapa, livres, K, origem, instancia):
    """Decodifica a bitstring quântica em uma sequência de visitas.

    Em caso de posições/clientes duplicados ou ausentes (comum em QAOA com
    poucas camadas/amostras), o reparo NÃO é mais arbitrário: a cada gap,
    escolhe-se, entre os clientes ainda não usados, aquele que resulta no
    MENOR tempo de chegada (heurística gulosa "vizinho mais viável no
    tempo"), em vez de simplesmente pegar o primeiro da lista. Isso evita
    entregar ao pós-processamento um ponto de partida artificialmente ruim.
    """
    bits = [int(c) for c in bitstring]

    atribuicoes = {}  # posicao -> lista de clientes candidatos (bit=1)
    for var_id, (cliente, k) in mapa.items():
        if bits[var_id] == 1:
            atribuicoes.setdefault(k, []).append(cliente)

    A, servico, C = instancia["A"], instancia["servico"], instancia["C"]
    usados = set()
    sequencia = []
    no_anterior = origem
    t_atual = A[origem]

    for k in range(1, K + 1):
        candidatos = [c for c in atribuicoes.get(k, []) if c not in usados]
        if not candidatos:
            candidatos = [c for c in livres if c not in usados]

        # escolhe o candidato com menor tempo de chegada resultante
        def chegada(c):
            return max(t_atual + servico[no_anterior] + C[no_anterior, c], A[c])

        escolhido = min(candidatos, key=chegada)
        sequencia.append(escolhido)
        usados.add(escolhido)
        t_atual = chegada(escolhido)
        no_anterior = escolhido

    return sequencia


def custo_rota(seq_completa, C):
    return sum(C[seq_completa[i], seq_completa[i + 1]] for i in range(len(seq_completa) - 1))


def tempos_de_chegada(seq_completa, instancia):
    A, servico, C = instancia["A"], instancia["servico"], instancia["C"]
    t = [max(A[seq_completa[0]], 0.0)]
    for i in range(1, len(seq_completa)):
        anterior, atual = seq_completa[i - 1], seq_completa[i]
        chegada = t[-1] + servico[anterior] + C[anterior, atual]
        t.append(max(chegada, A[atual]))
    return t


# ---------------------------------------------------------------------------
# Simulação do percurso real (execução "até onde der")
# ---------------------------------------------------------------------------

def simular_percurso(rota, instancia, fechar_no_deposito=False):
    """Executa a rota nó a nó com o relógio real, partindo do depósito.

    A cada perna, calcula-se a chegada em j como
        chegada_j = max(t_atual + servico_i + c_ij, a_j)
    Se chegada_j > b_j, o veículo NÃO consegue atender j dentro da janela:
    o percurso é interrompido nesse ponto. Nada é descartado antes da
    execução — a rota é sempre tentada até onde for possível.

    Retorna um dicionário com:
      - 'percorrido'   : lista dos nós efetivamente visitados, na ordem, do
                         depósito até o último nó atendido dentro da janela;
      - 'nao_visitados': lista dos nós da rota que não puderam ser visitados
                         (a partir do primeiro que estourou a janela);
      - 'inviavel'     : True se algum nó ficou de fora (ou se o retorno ao
                         depósito estourou b_0, quando fechar_no_deposito);
      - 'tempos'       : tempos de chegada nos nós percorridos;
      - 'no_falha'     : o nó em que o percurso foi interrompido (ou None);
      - 'chegada_falha': o tempo de chegada que estourou a janela (ou None);
      - 'retorno_deposito': tempo de retorno ao depósito, se aplicável.
    """
    A, B = instancia["A"], instancia["B"]
    servico, C = instancia["servico"], instancia["C"]

    inicio = rota[0]
    percorrido = [inicio]
    tempos = [float(A[inicio])]
    t_atual = tempos[0]
    atual = inicio

    nao_visitados = []
    inviavel = False
    no_falha = None
    chegada_falha = None
    retorno_deposito = None

    for pos in range(1, len(rota)):
        prox = rota[pos]
        chegada = max(t_atual + servico[atual] + C[atual, prox], A[prox])
        if chegada > B[prox]:
            inviavel = True
            no_falha = prox
            chegada_falha = float(chegada)
            nao_visitados = list(rota[pos:])
            break
        percorrido.append(prox)
        tempos.append(float(chegada))
        t_atual, atual = chegada, prox

    if fechar_no_deposito and not inviavel:
        retorno = t_atual + servico[atual] + C[atual, inicio]
        retorno_deposito = float(retorno)
        if retorno > B[inicio]:
            inviavel = True
            no_falha = inicio
            chegada_falha = float(retorno)
            nao_visitados = [inicio]  # não dá para voltar ao depósito no prazo

    return dict(
        percorrido=percorrido,
        nao_visitados=nao_visitados,
        inviavel=inviavel,
        tempos=tempos,
        no_falha=no_falha,
        chegada_falha=chegada_falha,
        retorno_deposito=retorno_deposito,
    )


def viola_janela(seq_completa, instancia):
    """True se a sequência não pode ser executada inteira dentro das
    janelas de tempo (usa a mesma simulação do percurso real)."""
    return simular_percurso(seq_completa, instancia)["inviavel"]


def busca_local_global(rota_completa, instancia):
    """Repáro final sobre a ROTA GLOBAL já montada (todas as fatias
    concatenadas), usando os tempos de chegada REAIS e acumulados desde o
    depósito.

    IMPORTANTE: esta versão NÃO descarta nem prioriza a eliminação de
    sub-rotas/candidatos inviáveis. O critério de aceitação do 2-opt/Or-opt
    é puramente a DISTÂNCIA total (custo_rota); uma solução inviável
    (com violação de janela de tempo) pode permanecer como resultado final
    — ela só é sinalizada (inviavel_global) para fins de relatório, mas não
    é eliminada nem penalizada durante a busca."""
    depósito = rota_completa[0]
    seq = list(rota_completa) + [depósito]  # fecha o ciclo no depósito
    C = instancia["C"]
    melhor = seq
    melhor_custo = custo_rota(melhor, C)
    melhorou = True

    while melhorou:
        melhorou = False
        for i in range(1, len(melhor) - 2):
            for j in range(i + 1, len(melhor) - 1):
                candidato = melhor[:i] + melhor[i : j + 1][::-1] + melhor[j + 1 :]
                c = custo_rota(candidato, C)
                if c < melhor_custo:
                    melhor, melhor_custo = candidato, c
                    melhorou = True
        for i in range(1, len(melhor) - 1):
            cliente = melhor[i]
            sem_cliente = melhor[:i] + melhor[i + 1 :]
            for pos in range(1, len(sem_cliente) - 1):
                candidato = sem_cliente[:pos] + [cliente] + sem_cliente[pos:]
                if candidato == melhor or candidato[0] != depósito or candidato[-1] != depósito:
                    continue
                c = custo_rota(candidato, C)
                if c < melhor_custo:
                    melhor, melhor_custo = candidato, c
                    melhorou = True

    inviavel_global = viola_janela(melhor, instancia)
    return melhor[:-1], inviavel_global  # remove o depósito duplicado no fim


def busca_local_restrita(sub_rota_quantica, origem, instancia):
    """Busca local de melhor-melhora, restrita à sub-rota, combinando dois
    tipos de vizinhança:

    - 2-opt: inverte um segmento (bom para reduzir distância).
    - Or-opt: realoca um único cliente para outra posição.

    IMPORTANTE: esta versão NÃO elimina/descarta sub-rotas que tornam a
    solução inviável. O critério de aceitação é puramente a DISTÂNCIA
    (custo_rota) — não há mais priorização de viabilidade (nº de
    violações/atraso) sobre custo. Se a sub-rota resultante for inviável
    (viola alguma janela de tempo), ela é apenas sinalizada em 'inviavel',
    mas mantida como está."""
    melhor = [origem] + list(sub_rota_quantica)
    C = instancia["C"]
    melhor_custo = custo_rota(melhor, C)
    melhorou = True

    while melhorou:
        melhorou = False

        # --- vizinhança 2-opt (inversão de segmento) ---
        for i in range(1, len(melhor) - 1):
            for j in range(i + 1, len(melhor)):
                candidato = melhor[:i] + melhor[i : j + 1][::-1] + melhor[j + 1 :]
                c = custo_rota(candidato, C)
                if c < melhor_custo:
                    melhor, melhor_custo = candidato, c
                    melhorou = True

        # --- vizinhança Or-opt (realocação de 1 cliente) ---
        for i in range(1, len(melhor)):
            cliente = melhor[i]
            sem_cliente = melhor[:i] + melhor[i + 1 :]
            for pos in range(1, len(sem_cliente) + 1):
                candidato = sem_cliente[:pos] + [cliente] + sem_cliente[pos:]
                if candidato == melhor:
                    continue
                c = custo_rota(candidato, C)
                if c < melhor_custo:
                    melhor, melhor_custo = candidato, c
                    melhorou = True

    inviavel = viola_janela(melhor, instancia)
    return melhor[1:], inviavel  # remove a origem fixa antes de retornar


# ---------------------------------------------------------------------------
# 4.3.6 Análise: baseline clássico (2-opt puro) para comparação (Gap %)
# ---------------------------------------------------------------------------

def baseline_classico_2opt(rota_inicial, instancia):
    C = instancia["C"]
    seq = list(rota_inicial) + [rota_inicial[0]]  # ciclo fechado no depósito
    melhorou = True
    while melhorou:
        melhorou = False
        for i in range(1, len(seq) - 2):
            for j in range(i + 1, len(seq) - 1):
                candidato = seq[:i] + seq[i : j + 1][::-1] + seq[j + 1 :]
                if custo_rota(candidato, C) < custo_rota(seq, C):
                    seq, melhorou = candidato, True
    return seq


# ---------------------------------------------------------------------------
# Pipeline principal (Quadro 6 do TCC)
# ---------------------------------------------------------------------------

def pipeline_hibrido(instancia, K=4, camadas_qaoa=1, amostras=200, maxiter=15, n_clientes=12):
    print("=" * 70)
    print("PIPELINE HÍBRIDO QUÂNTICO-CLÁSSICO PARA O TSPTW (demo)")
    print("=" * 70)

    # 1) Carregamento das instâncias: recorte direto dos n_clientes
    # primeiros clientes (sem nenhum pré-filtro de viabilidade).
    origem_instancia = instancia
    if isinstance(origem_instancia, str) and _parece_caminho(origem_instancia):
        print(f"\n[1] Arquivo: {resolver_caminho_instancia(origem_instancia)}")

    instancia = carregar_instancia(instancia, n_clientes=n_clientes)
    print(
        f"[1] Instância carregada: {instancia['n']} nós "
        f"(depósito + {instancia['n'] - 1} clientes, recorte direto por ID)"
    )

    # 2) Rota inicial (heurística) + Path-Slicing
    rota_inicial = gerar_rota_heuristica_chegada_mais_proxima(instancia)
    fatias = path_slicing(rota_inicial, K)
    print(f"[2] Rota inicial (heurística): {rota_inicial}")
    print(f"    Path-slicing (K={K}): {fatias}")

    rota_global = [rota_inicial[0]]
    tempo_classico_total = 0.0
    tempo_quantico_total = 0.0
    alguma_inviavel = False

    for idx_fatia, fatia in enumerate(fatias, start=1):
        print(f"\n--- Sub-rota {idx_fatia}/{len(fatias)}: {fatia} ---")
        t0 = time_module.time()

        # 3) Tradutor QUBO
        Q, n_vars, mapa, origem = construir_qubo(fatia, instancia)
        h, Jm, offset = qubo_para_ising(Q, n_vars)
        print(f"    QUBO construído: {n_vars} variáveis binárias")
        tempo_classico_total += time_module.time() - t0

        # 4) Execução quântica (QAOA no LocalSimulator)
        bitstring, t_sim = executar_qaoa(
            n_vars, h, Jm, offset, camadas=camadas_qaoa,
            amostras=amostras, maxiter=maxiter,
        )
        tempo_quantico_total += t_sim
        print(f"    Bitstring ótima (QAOA): {bitstring}")

        # 5) Pós-processamento (decodificação + 2-opt restrito)
        t0 = time_module.time()
        livres = fatia[1:]
        K_fatia = len(livres)
        sub_rota_quantica = decodificar_bitstring(
            bitstring, mapa, livres, K_fatia, origem, instancia
        )
        sub_rota_refinada, inviavel = busca_local_restrita(
            sub_rota_quantica, origem, instancia
        )
        tempo_classico_total += time_module.time() - t0

        if inviavel:
            alguma_inviavel = True
            print("    [!] Sub-rota sinalizada como INVIÁVEL (violação de janela) — mantida como está.")

        print(f"    Sub-rota refinada (2-opt): {sub_rota_refinada}")
        rota_global.extend(sub_rota_refinada)

    # 5.5) Reconciliação global sobre a rota inteira, com o relógio real.
    # Assim como nas sub-rotas, violações de janela NÃO são eliminadas —
    # apenas sinalizadas em inviavel_global.
    t0 = time_module.time()
    rota_global, inviavel_global = busca_local_global(rota_global, instancia)
    tempo_classico_total += time_module.time() - t0

    # 5.6) Execução do percurso "até onde der": o veículo tenta seguir a
    # rota planejada mesmo que ela seja inviável, e para no primeiro nó
    # cuja janela de tempo não pode ser respeitada.
    percurso = simular_percurso(rota_global, instancia, fechar_no_deposito=True)
    inviavel_global = percurso["inviavel"]

    # caminho exibido: se tudo deu certo, inclui o retorno ao depósito
    caminho_percorrido = list(percurso["percorrido"])
    if not inviavel_global:
        caminho_percorrido = caminho_percorrido + [rota_global[0]]

    # 6) Análise: custo final, tempos e comparação com baseline clássico
    custo_hibrido = custo_rota(rota_global + [rota_global[0]], instancia["C"])
    custo_percorrido = custo_rota(caminho_percorrido, instancia["C"])
    baseline = baseline_classico_2opt(rota_inicial, instancia)
    custo_classico = custo_rota(baseline, instancia["C"])
    gap = (custo_hibrido - custo_classico) / custo_classico * 100

    tte = tempo_classico_total + tempo_quantico_total

    print("\n" + "=" * 70)
    print("ANÁLISE DE RESULTADOS")
    print("=" * 70)
    print(f"Rota global planejada:     {rota_global}")
    print(f"Custo da rota híbrida:     {custo_hibrido:.2f}")
    print(f"Custo do baseline (2-opt): {custo_classico:.2f}")
    print(f"Gap (%) vs. baseline:      {gap:.2f}%")

    print("\n" + "-" * 70)
    print("EXECUÇÃO DO PERCURSO (tentativa nó a nó, com o relógio real)")
    print("-" * 70)
    print(f"Situação da solução:       {'INVIÁVEL' if inviavel_global else 'VIÁVEL'}")
    print(f"Caminho percorrido:        {caminho_percorrido}")
    print(
        "Tempos de chegada:         "
        + ", ".join(
            f"{no}@{t:.1f}" for no, t in zip(percurso["percorrido"], percurso["tempos"])
        )
    )
    print(f"Último nó visitado:        {percurso['percorrido'][-1]}")
    print(f"Distância percorrida:      {custo_percorrido:.2f}")
    if inviavel_global:
        print(
            f"Parou em:                  nó {percurso['no_falha']} "
            f"(chegada {percurso['chegada_falha']:.1f} > "
            f"b_{percurso['no_falha']} = {instancia['B'][percurso['no_falha']]:.1f})"
        )
        print(f"Nós NÃO visitados:         {percurso['nao_visitados']}")
        print(f"Qtd. de nós não visitados: {len(percurso['nao_visitados'])}")
    else:
        print("Nós NÃO visitados:         [] (todos os nós foram atendidos)")
        print(f"Retorno ao depósito em:    {percurso['retorno_deposito']:.1f}")

    print("\n" + "-" * 70)
    print(f"Tempo Clássico (TCL):      {tempo_classico_total:.3f} s")
    print(f"Tempo Simulação Quântica (TSQ): {tempo_quantico_total:.3f} s")
    print(f"Tempo Total de Execução (TTE):  {tte:.3f} s")

    return dict(
        rota_global=rota_global,
        caminho_percorrido=caminho_percorrido,
        nao_visitados=percurso["nao_visitados"],
        ultimo_no_visitado=percurso["percorrido"][-1],
        no_falha=percurso["no_falha"],
        chegada_falha=percurso["chegada_falha"],
        tempos_chegada=percurso["tempos"],
        custo_hibrido=custo_hibrido,
        custo_percorrido=custo_percorrido,
        custo_classico=custo_classico,
        gap=gap,
        inviavel=inviavel_global,
        tcl=tempo_classico_total,
        tsq=tempo_quantico_total,
        tte=tte,
    )


if __name__ == "__main__":
    pipeline_hibrido(
        instancia=INSTANCIA_PADRAO,   # caminho do arquivo (ou o texto da instância)
        K=4,
        camadas_qaoa=1,
        amostras=200,
        maxiter=15,
        n_clientes=4,
    )