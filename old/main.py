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

Requisitos:
    pip install amazon-braket-sdk numpy scipy networkx
"""

import io
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

# Recorte da instância R101 de Solomon (1987), conforme Tabela 2 do TCC,
# adaptada para veículo único com capacidade infinita (colunas:
# nó, x, y, tempo mínimo a_i, tempo máximo b_i, tempo de serviço).
R101_SUBSET_TXT = """
0  35.00 35.00   0 230  0
1  41.00 49.00 161 171 10
2  35.00 17.00  50  60 10
3  55.00 45.00 116 126 10
4  55.00 20.00 149 159 10
5  15.00 30.00  34  44 10
6  25.00 30.00  99 109 10
7  20.00 50.00  81  91 10
8  10.00 43.00  95 105 10
9  55.00 60.00  97 107 10
10 30.00 60.00 124 134 10
11 20.00 65.00  67  77 10
12 50.00 35.00  63  73 10
"""


def carregar_instancia(texto=R101_SUBSET_TXT):
    """Realiza a ingestão, parsing e validação estrutural dos dados do
    benchmark de Solomon (1987), isolando coordenadas (Xi, Yi), tempo de
    serviço e janelas de tempo [ai, bi]. Produz a matriz de distâncias
    euclidianas C e os vetores A (mínimos) e B (máximos)."""
    nos = []
    for linha in io.StringIO(texto.strip()).readlines():
        partes = linha.split()
        if not partes:
            continue
        idx, x, y, a, b, serv = partes
        nos.append(
            dict(
                id=int(idx),
                x=float(x),
                y=float(y),
                a=float(a),
                b=float(b),
                servico=float(serv),
            )
        )

    n = len(nos)
    C = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        dx = nos[i]["x"] - nos[j]["x"]
        dy = nos[i]["y"] - nos[j]["y"]
        C[i, j] = math.sqrt(dx * dx + dy * dy)

    A = np.array([no["a"] for no in nos])
    B = np.array([no["b"] for no in nos])
    servico = np.array([no["servico"] for no in nos])

    # Validação (assert) da relação ai <= bi para todos os vértices
    assert np.all(A <= B), "Instância inválida: existe ai > bi."

    return dict(nos=nos, C=C, A=A, B=B, servico=servico, n=n)


def gerar_rota_heuristica_chegada_mais_proxima(instancia):
    """Heurística gulosa: ordena os clientes pelo início da janela de tempo
    (ai), como descrito na Seção 4.4.1 do TCC, mantendo o depósito v0 como
    origem."""
    nos = instancia["nos"]
    depósito = nos[0]
    demais = sorted(nos[1:], key=lambda no: no["a"])
    rota = [depósito["id"]] + [no["id"] for no in demais]
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

def decodificar_bitstring(bitstring, mapa, livres, K):
    """Decodifica a bitstring quântica em uma sequência de visitas.
    Em caso de posições/clientes duplicados (comum em QAOA com poucas
    camadas/amostras), aplica um reparo guloso simples."""
    bits = [int(c) for c in bitstring]

    atribuicoes = {}  # posicao -> lista de clientes candidatos (bit=1)
    for var_id, (cliente, k) in mapa.items():
        if bits[var_id] == 1:
            atribuicoes.setdefault(k, []).append(cliente)

    usados = set()
    sequencia = []
    for k in range(1, K + 1):
        candidatos = [c for c in atribuicoes.get(k, []) if c not in usados]
        if candidatos:
            escolhido = candidatos[0]
        else:
            restantes = [c for c in livres if c not in usados]
            escolhido = restantes[0] if restantes else livres[0]
        sequencia.append(escolhido)
        usados.add(escolhido)

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


def viola_janela(seq_completa, instancia):
    B = instancia["B"]
    t = tempos_de_chegada(seq_completa, instancia)
    return any(t[i] > B[seq_completa[i]] for i in range(len(seq_completa)))


def busca_local_2opt_restrito(sub_rota_quantica, origem, instancia):
    """Busca local 2-opt (melhor melhora), restrita à sub-rota, corrigindo
    violações residuais de janela de tempo, conforme Seção 4.4.2."""
    C = instancia["C"]
    melhor = [origem] + list(sub_rota_quantica)
    melhorou = True

    def custo_com_penalidade(seq):
        custo = custo_rota(seq, C)
        if viola_janela(seq, instancia):
            custo += 1e4  # penalidade forte por inviabilidade
        return custo

    melhor_custo = custo_com_penalidade(melhor)

    while melhorou:
        melhorou = False
        for i in range(1, len(melhor) - 1):
            for j in range(i + 1, len(melhor)):
                candidato = melhor[:i] + melhor[i : j + 1][::-1] + melhor[j + 1 :]
                c = custo_com_penalidade(candidato)
                if c < melhor_custo:
                    melhor, melhor_custo = candidato, c
                    melhorou = True
        # loop continua até nenhum movimento melhorar (ótimo local)

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

def pipeline_hibrido(K=4, camadas_qaoa=1, amostras=200, maxiter=15):
    print("=" * 70)
    print("PIPELINE HÍBRIDO QUÂNTICO-CLÁSSICO PARA O TSPTW (demo)")
    print("=" * 70)

    # 1) Carregamento das instâncias
    instancia = carregar_instancia()
    print(f"\n[1] Instância carregada: {instancia['n']} nós (recorte de R101)")

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
        sub_rota_quantica = decodificar_bitstring(bitstring, mapa, livres, K_fatia)
        sub_rota_refinada, inviavel = busca_local_2opt_restrito(
            sub_rota_quantica, origem, instancia
        )
        tempo_classico_total += time_module.time() - t0

        if inviavel:
            alguma_inviavel = True
            print("    [!] Sub-rota sinalizada como INVIÁVEL (violação de janela).")

        print(f"    Sub-rota refinada (2-opt): {sub_rota_refinada}")
        rota_global.extend(sub_rota_refinada)

    # 6) Análise: custo final, tempos e comparação com baseline clássico
    custo_hibrido = custo_rota(rota_global + [rota_global[0]], instancia["C"])
    baseline = baseline_classico_2opt(rota_inicial, instancia)
    custo_classico = custo_rota(baseline, instancia["C"])
    gap = (custo_hibrido - custo_classico) / custo_classico * 100

    tte = tempo_classico_total + tempo_quantico_total

    print("\n" + "=" * 70)
    print("ANÁLISE DE RESULTADOS")
    print("=" * 70)
    print(f"Rota global (híbrida):     {rota_global}")
    print(f"Custo da rota híbrida:     {custo_hibrido:.2f}")
    print(f"Custo do baseline (2-opt): {custo_classico:.2f}")
    print(f"Gap (%) vs. baseline:      {gap:.2f}%")
    print(f"Viabilidade temporal:      {'INVIÁVEL' if alguma_inviavel else 'VIÁVEL'}")
    print(f"Tempo Clássico (TCL):      {tempo_classico_total:.3f} s")
    print(f"Tempo Simulação Quântica (TSQ): {tempo_quantico_total:.3f} s")
    print(f"Tempo Total de Execução (TTE):  {tte:.3f} s")

    return dict(
        rota_global=rota_global,
        custo_hibrido=custo_hibrido,
        custo_classico=custo_classico,
        gap=gap,
        inviavel=alguma_inviavel,
        tcl=tempo_classico_total,
        tsq=tempo_quantico_total,
        tte=tte,
    )


if __name__ == "__main__":
    pipeline_hibrido(K=4, camadas_qaoa=1, amostras=200, maxiter=15)