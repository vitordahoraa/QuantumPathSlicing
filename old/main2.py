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

# Instância R101 de Solomon (1987) completa (101 nós: depósito + 100
# clientes), conforme Tabela 2 do TCC, adaptada para veículo único com
# capacidade infinita (colunas: nó, x, y, tempo mínimo a_i, tempo máximo
# b_i, tempo de serviço).
RC206 = """
0  0.0000   0.0000   0   960   0
1  43.0116  0.0000   43  283  10
2  33.2303  13.9910 36  276  10
3  29.3687  16.2011 33  273  10
"""
R101_FULL_TXT = """
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
13 30.00 25.00 159 169 10
14 15.00 10.00  32  42 10
15 30.00  5.00  61  71 10
16 10.00 20.00  75  85 10
17  5.00 30.00 157 167 10
18 20.00 40.00  87  97 10
19 15.00 60.00  76  86 10
20 45.00 65.00 126 136 10
21 45.00 20.00  62  72 10
22 45.00 10.00  97 107 10
23 55.00  5.00  68  78 10
24 65.00 35.00 153 163 10
25 65.00 20.00 172 182 10
26 45.00 30.00 132 142 10
27 35.00 40.00  37  47 10
28 41.00 37.00  39  49 10
29 64.00 42.00  63  73 10
30 40.00 60.00  71  81 10
31 31.00 52.00  50  60 10
32 35.00 69.00 141 151 10
33 53.00 52.00  37  47 10
34 65.00 55.00 117 127 10
35 63.00 65.00 143 153 10
36  2.00 60.00  41  51 10
37 20.00 20.00 134 144 10
38  5.00  5.00  83  93 10
39 60.00 12.00  44  54 10
40 40.00 25.00  85  95 10
41 42.00  7.00  97 107 10
42 24.00 12.00  31  41 10
43 23.00  3.00 132 142 10
44 11.00 14.00  69  79 10
45  6.00 38.00  32  42 10
46  2.00 48.00 117 127 10
47  8.00 56.00  51  61 10
48 13.00 52.00 165 175 10
49  6.00 68.00 108 118 10
50 47.00 47.00 124 134 10
51 49.00 58.00  88  98 10
52 27.00 43.00  52  62 10
53 37.00 31.00  95 105 10
54 57.00 29.00 140 150 10
55 63.00 23.00 136 146 10
56 53.00 12.00 130 140 10
57 32.00 12.00 101 111 10
58 36.00 26.00 200 210 10
59 21.00 24.00  18  28 10
60 17.00 34.00 162 172 10
61 12.00 24.00  76  86 10
62 24.00 58.00  58  68 10
63 27.00 69.00  34  44 10
64 15.00 77.00  73  83 10
65 62.00 77.00  51  61 10
66 49.00 73.00 127 137 10
67 67.00  5.00  83  93 10
68 56.00 39.00 142 152 10
69 37.00 47.00  50  60 10
70 37.00 56.00 182 192 10
71 57.00 68.00  77  87 10
72 47.00 16.00  35  45 10
73 44.00 17.00  78  88 10
74 46.00 13.00 149 159 10
75 49.00 11.00  69  79 10
76 49.00 42.00  73  83 10
77 53.00 43.00 179 189 10
78 61.00 52.00  96 106 10
79 57.00 48.00  92 102 10
80 56.00 37.00 182 192 10
81 55.00 54.00  94 104 10
82 15.00 47.00  55  65 10
83 14.00 37.00  44  54 10
84 11.00 31.00 101 111 10
85 16.00 22.00  91 101 10
86  4.00 18.00  94 104 10
87 28.00 18.00  93 103 10
88 26.00 52.00  74  84 10
89 26.00 35.00 176 186 10
90 31.00 67.00  95 105 10
91 15.00 19.00 160 170 10
92 22.00 22.00  18  28 10
93 18.00 24.00 188 198 10
94 26.00 27.00 100 110 10
95 25.00 24.00  39  49 10
96 22.00 27.00 135 145 10
97 25.00 21.00 133 143 10
98 19.00 21.00  58  68 10
99 20.00 26.00  83  93 10
100 18.00 18.00 185 195 10
"""


def selecionar_subconjunto_proximo(instancia_completa, n_clientes):
    """Constrói um subconjunto de n_clientes que é VIÁVEL POR CONSTRUÇÃO,
    simulando um veículo percorrendo a instância COMPLETA (100 clientes) e
    aceitando, a cada passo, apenas o próximo cliente cuja chegada real
    (considerando deslocamento + tempo de serviço acumulado) respeita a
    janela [ai, bi]. Clientes que tornariam a chegada inviável são
    simplesmente pulados (não entram no subconjunto).

    Por que não bastava filtrar por proximidade geográfica nem por janelas
    "escalonadas" isoladamente: no grupo R (aleatório) do benchmark de
    Solomon, (i) clientes próximos no espaço podem ter janelas quase
    idênticas e sobrepostas — impossível atender ambos com um único
    veículo, pois o tempo de serviço de um já consome a janela do outro; e
    (ii) clientes com janelas bem espaçadas no tempo podem estar longe o
    suficiente no espaço para que o deslocamento entre eles, sozinho, já
    estoure a folga disponível. Somente simulando a viagem real (tempo +
    distância) é possível garantir, por construção, que o subconjunto
    resultante admite ao menos uma ordem de visita totalmente viável."""
    nos = instancia_completa["nos"]
    depósito = nos[0]
    C_completa = instancia_completa["C"]
    A_completa = instancia_completa["A"]
    B_completa = instancia_completa["B"]
    servico_completa = instancia_completa["servico"]

    restantes = {no["id"] for no in nos[1:]}
    selecionados_ids = []
    t_atual = A_completa[0]
    atual = 0

    while restantes and len(selecionados_ids) < n_clientes:
        melhor = None
        for c in restantes:
            chegada = max(
                t_atual + servico_completa[atual] + C_completa[atual, c],
                A_completa[c],
            )
            if chegada <= B_completa[c]:
                if melhor is None or chegada < melhor[0]:
                    melhor = (chegada, c)
        if melhor is None:
            # nenhum cliente restante é alcançável em tempo hábil a partir
            # do ponto atual; encerra a construção do subconjunto aqui
            break
        chegada, escolhido = melhor
        selecionados_ids.append(escolhido)
        restantes.discard(escolhido)
        t_atual, atual = chegada, escolhido

    if len(selecionados_ids) < n_clientes:
        print(
            f"    [aviso] A construção viável só conseguiu encadear "
            f"{len(selecionados_ids)} de {n_clientes} clientes solicitados "
            f"a partir do depósito, sem violar nenhuma janela; usando "
            f"apenas esses {len(selecionados_ids)}."
        )

    id_para_no = {no["id"]: no for no in nos}
    selecionados = [id_para_no[i] for i in selecionados_ids]
    return _reconstruir_instancia([depósito] + selecionados)


def _reconstruir_instancia(nos_selecionados):
    """Reindexa os nós selecionados (0..n-1) e recalcula C, A, B a partir
    da lista filtrada, mantendo o formato usado pelo restante do pipeline."""
    n = len(nos_selecionados)
    C = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        dx = nos_selecionados[i]["x"] - nos_selecionados[j]["x"]
        dy = nos_selecionados[i]["y"] - nos_selecionados[j]["y"]
        C[i, j] = math.sqrt(dx * dx + dy * dy)

    nos_reindexados = [
        dict(no, id=novo_id) for novo_id, no in enumerate(nos_selecionados)
    ]
    A = np.array([no["a"] for no in nos_reindexados])
    B = np.array([no["b"] for no in nos_reindexados])
    servico = np.array([no["servico"] for no in nos_reindexados])
    assert np.all(A <= B), "Instância inválida: existe ai > bi."
    return dict(nos=nos_reindexados, C=C, A=A, B=B, servico=servico, n=n)


def carregar_instancia(texto=R101_FULL_TXT, n_clientes=None):
    """Realiza a ingestão, parsing e validação estrutural dos dados do
    benchmark de Solomon (1987), isolando coordenadas (Xi, Yi), tempo de
    serviço e janelas de tempo [ai, bi]. Produz a matriz de distâncias
    euclidianas C e os vetores A (mínimos) e B (máximos).

    Se n_clientes for informado, em vez de usar a instância completa (100
    clientes), seleciona os n_clientes GEOGRAFICAMENTE MAIS PRÓXIMOS do
    depósito (ver selecionar_subconjunto_proximo), o que produz recortes
    muito mais propensos a admitir soluções viáveis do que pegar os N
    primeiros nós pelo ID."""
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

    instancia_completa = _reconstruir_instancia(nos)

    if n_clientes is not None:
        return selecionar_subconjunto_proximo(instancia_completa, n_clientes)
    return instancia_completa


def gerar_rota_heuristica_chegada_mais_proxima(instancia):
    """Heurística construtiva gulosa "vizinho mais próximo com viabilidade
    de janela" (nearest neighbor time-feasible), conforme Seção 4.4.1 do
    TCC. A cada passo, entre os clientes ainda não visitados, escolhe-se o
    que resulta no MENOR tempo de chegada a partir do nó atual, priorizando
    os que ainda são fisicamente viáveis (chegada <= bi); se nenhum for
    viável, escolhe-se o de menor atraso.

    Isso substitui a simples ordenação por ai: ordenar só pelo início da
    janela ignora a geografia por completo e pode agrupar clientes distantes
    entre si com janelas apertadas, tornando QUALQUER ordem de visita
    inviável (independentemente do algoritmo de busca local usado depois).
    A construção aqui já busca uma sequência espacialmente coerente e
    cronologicamente viável desde o início."""
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


def viola_janela(seq_completa, instancia):
    B = instancia["B"]
    t = tempos_de_chegada(seq_completa, instancia)
    return any(t[i] > B[seq_completa[i]] for i in range(len(seq_completa)))


def custo_lexicografico(seq, instancia):
    """Retorna uma tupla (nº de violações, atraso total, distância total),
    comparada lexicograficamente. Isso separa o objetivo de VIABILIDADE do
    objetivo de CUSTO, em vez de somá-los numa única penalidade escalar
    (o que fazia o 2-opt puro trocar viabilidade por poucos km de economia
    ou vice-versa de forma pouco controlada)."""
    B = instancia["B"]
    C = instancia["C"]
    t = tempos_de_chegada(seq, instancia)
    n_violacoes = sum(1 for i in range(len(seq)) if t[i] > B[seq[i]])
    atraso_total = sum(max(0.0, t[i] - B[seq[i]]) for i in range(len(seq)))
    distancia = custo_rota(seq, C)
    return (n_violacoes, atraso_total, distancia)


def busca_local_global(rota_completa, instancia):
    """Repáro final sobre a ROTA GLOBAL já montada (todas as fatias
    concatenadas), usando os tempos de chegada REAIS e acumulados desde o
    depósito — ao contrário da busca local por sub-rota, que assume
    (de forma otimista) que a sub-rota começa exatamente no início da
    janela do nó de origem (ai), ignorando o horário real de chegada vindo
    da fatia anterior.

    Esse passo existe porque o path-slicing resolve cada subproblema de
    forma independente (Seção 4.3.2/4.7 do TCC): uma sub-rota pode ser
    localmente viável assumindo partida em ai, mas a chegada real ao nó de
    origem (herdada da fatia anterior) costuma ser mais tarde, quebrando a
    viabilidade quando as fatias são concatenadas. Aqui aplicamos a mesma
    vizinhança 2-opt + Or-opt com critério lexicográfico, mas sobre a rota
    inteira e com o relógio real, funcionando como uma reconciliação global
    entre as fatias."""
    depósito = rota_completa[0]
    seq = list(rota_completa) + [depósito]  # fecha o ciclo no depósito
    melhor = seq
    melhor_custo = custo_lexicografico(melhor, instancia)
    melhorou = True

    while melhorou:
        melhorou = False
        for i in range(1, len(melhor) - 2):
            for j in range(i + 1, len(melhor) - 1):
                candidato = melhor[:i] + melhor[i : j + 1][::-1] + melhor[j + 1 :]
                c = custo_lexicografico(candidato, instancia)
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
                c = custo_lexicografico(candidato, instancia)
                if c < melhor_custo:
                    melhor, melhor_custo = candidato, c
                    melhorou = True

    inviavel_global = melhor_custo[0] > 0
    return melhor[:-1], inviavel_global  # remove o depósito duplicado no fim


def busca_local_restrita(sub_rota_quantica, origem, instancia):
    """Busca local de melhor-melhora, restrita à sub-rota, combinando dois
    tipos de vizinhança (Seção 4.4.2 estendida):

    - 2-opt: inverte um segmento (bom para reduzir distância).
    - Or-opt: realoca um único cliente para outra posição (bom para
      corrigir violações de janela de tempo, já que muda a ordem
      cronológica relativa sem inverter o resto do trecho).

    O critério de aceitação é LEXICOGRÁFICO: uma solução só substitui a
    melhor encontrada se tiver menos violações; em caso de empate, se tiver
    menor atraso total; em caso de novo empate, se tiver menor distância.
    Isso prioriza viabilidade antes de custo, o que o 2-opt puro (com
    penalidade aditiva) não garantia.
    """
    melhor = [origem] + list(sub_rota_quantica)
    melhor_custo = custo_lexicografico(melhor, instancia)
    melhorou = True

    while melhorou:
        melhorou = False

        # --- vizinhança 2-opt (inversão de segmento) ---
        for i in range(1, len(melhor) - 1):
            for j in range(i + 1, len(melhor)):
                candidato = melhor[:i] + melhor[i : j + 1][::-1] + melhor[j + 1 :]
                c = custo_lexicografico(candidato, instancia)
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
                c = custo_lexicografico(candidato, instancia)
                if c < melhor_custo:
                    melhor, melhor_custo = candidato, c
                    melhorou = True

    inviavel = melhor_custo[0] > 0
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

    # 1) Carregamento das instâncias: seleciona os n_clientes mais próximos
    # do depósito dentro da R101 completa (em vez dos N primeiros por ID),
    # reduzindo a chance de o recorte ser fisicamente inviável.
    instancia = carregar_instancia(instancia, n_clientes=n_clientes)
    print(
        f"\n[1] Instância carregada: {instancia['n']} nós "
        f"(subconjunto de R101 construído para admitir ao menos uma "
        f"ordem de visita viável)"
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
            print("    [!] Sub-rota sinalizada como INVIÁVEL (violação de janela).")

        print(f"    Sub-rota refinada (2-opt): {sub_rota_refinada}")
        rota_global.extend(sub_rota_refinada)

    # 5.5) Reconciliação global: repara violações que só aparecem quando as
    # fatias são concatenadas com o relógio real (ver busca_local_global)
    t0 = time_module.time()
    rota_global, inviavel_global = busca_local_global(rota_global, instancia)
    tempo_classico_total += time_module.time() - t0
    if inviavel_global:
        print("\n[!] Ainda há violação de janela após a reconciliação global.")
    else:
        print("\n[OK] Rota global reconciliada e VIÁVEL após o reparo final.")

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
    print(f"Viabilidade temporal (global): {'INVIÁVEL' if inviavel_global else 'VIÁVEL'}")
    print(f"Tempo Clássico (TCL):      {tempo_classico_total:.3f} s")
    print(f"Tempo Simulação Quântica (TSQ): {tempo_quantico_total:.3f} s")
    print(f"Tempo Total de Execução (TTE):  {tte:.3f} s")

    return dict(
        rota_global=rota_global,
        custo_hibrido=custo_hibrido,
        custo_classico=custo_classico,
        gap=gap,
        inviavel=inviavel_global,
        tcl=tempo_classico_total,
        tsq=tempo_quantico_total,
        tte=tte,
    )


if __name__ == "__main__":
    pipeline_hibrido(instancia=RC206,K=4, camadas_qaoa=1, amostras=200, maxiter=15, n_clientes=4)