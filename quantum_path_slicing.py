"""
quantum_path_slicing.py
===========================================================================
Abordagem híbrida quântica de busca local para o TSPTW, seguindo a
arquitetura do TCC:

    Path-Slicing -> Tradutor QUBO -> Execução Quântica (QAOA no Amazon
    Braket: LocalSimulator ou AWS) -> Pós-processamento (busca local) -> Análise

Toda a lógica QUÂNTICA vive neste módulo. A obtenção dos ângulos (gamma,
beta) do QAOA — a única parte com otimizador clássico — fica FORA dele, em
qaoa_calibracao.py: aqui só se monta o circuito com ângulos já definidos e
se mede. A avaliação de soluções
(distância, janelas de tempo, simulação do percurso) e as vizinhanças de
busca local são importadas de classic_algorithm.py, de modo que os dois
algoritmos sejam comparados sob EXATAMENTE a mesma função objetivo
lexicográfica (violação, distância).

---------------------------------------------------------------------------
NOTAS DE CALIBRAÇÃO (importantes para a leitura dos resultados)
---------------------------------------------------------------------------
1. PENALIDADES EM ESCALA. Em uma versão anterior, alpha e beta eram fixos
   em 8.0 enquanto os tempos de viagem da instância chegavam a ~73. Com
   isso H_Fluxo não conseguia impor a restrição one-hot: saía mais barato
   deixar variáveis em zero do que pagar o custo de viagem, e o mínimo do
   próprio QUBO NÃO era uma rota válida. Agora as penalidades são
   dimensionadas a partir da maior aresta da fatia (ver escala_penalidade),
   o que devolve ao QUBO a propriedade de ter permutações válidas como
   estados de menor energia.

2. DECODIFICAÇÃO FIEL. Antes, o decodificador usava apenas a bitstring de
   maior probabilidade e, como ela quase nunca era uma permutação válida,
   caía no reparo guloso — que é um vizinho-mais-próximo disfarçado. O
   efeito era que a saída do QAOA não influenciava o resultado final.
   Agora o decodificador recebe o RANKING de bitstrings medidas e escolhe
   a de maior probabilidade que seja uma permutação válida, recorrendo ao
   reparo só quando nenhuma das medidas é válida — e registrando isso.

3. MÉTRICAS DE QUALIDADE QUÂNTICA. Para fatias pequenas (n_vars <= 20) o
   espaço é enumerado por força bruta, permitindo medir onde a solução do
   QAOA cai no espectro de energias do QUBO (razão de aproximação e rank),
   e verificar se o ótimo do QUBO é mesmo uma permutação válida — ou seja,
   se a formulação está bem calibrada para aquela instância.
"""

import time as time_module

import numpy as np

from braket_dispositivo import criar_dispositivo
from classic_algorithm import (
    avaliar,
    custo_rota,
    rota_inicial as construir_rota_inicial,
    simular_percurso,
    tempos_de_chegada,
    vnd,
)

# O Braket só é necessário para o caminho quântico. A importação é
# tolerante para que `python main.py --algorithm classico` funcione em uma
# máquina sem o SDK instalado.
try:
    from braket.circuits import Circuit

    BRAKET_DISPONIVEL = True
    _ERRO_BRAKET = None
except ImportError as exc:  # pragma: no cover
    BRAKET_DISPONIVEL = False
    _ERRO_BRAKET = exc
    Circuit = None


# número máximo de variáveis para o qual vale enumerar o QUBO por força
# bruta (2^20 ~ 1e6 estados); acima disso as métricas exatas são puladas
LIMITE_FORCA_BRUTA = 20


def _exigir_braket():
    if not BRAKET_DISPONIVEL:
        raise ImportError(
            "O Amazon Braket SDK é necessário para o algoritmo quântico.\n"
            "Instale com:  pip install amazon-braket-sdk\n"
            f"Erro original: {_ERRO_BRAKET}"
        )


# ---------------------------------------------------------------------------
# 4.3.2 Path-Slicing
# ---------------------------------------------------------------------------

def path_slicing(rota, K):
    """Decompõe a rota global em subproblemas de tamanho fixo K, com
    sobreposição de 1 nó entre fatias consecutivas (o último nó da fatia M
    é a origem fixa da fatia M+1), conforme Seção 4.3.2/4.4.1 do TCC."""
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

def analise_temporal(sub_rota, instancia, t_inicial=None):
    """Análise temporal EXATA da fatia, por programação dinâmica sobre
    subconjuntos (estilo Held-Karp), usada para montar o H_Tempo.

    A versão anterior estimava o tempo até a posição k como
    "partida + k x menor_aresta + (k-1) x menor_serviço". Válido como
    limitante inferior, mas tão frouxo que quase nunca disparava: numa
    fatia real, 4 das 6 ordens estouravam a janela do destino e NENHUMA
    foi penalizada. O QAOA ficava livre para escolher uma delas.

    Aqui os limitantes são os valores mínimos de fato alcançáveis:

      D[S][i]    = chegada mais cedo em i tendo visitado exatamente S,
                   com i por último, partindo da origem no relógio real;
      Comp[S][i] = custo aditivo mínimo de i até o destino, passando pelos
                   que faltam (ignora espera, o que o mantém limitante
                   inferior).

    Disso saem duas famílias de proibições, ambas exatas no sentido de que
    só proíbem o que é comprovadamente impossível:

      - LINEARES  : o cliente i não cabe na posição k, seja porque não dá
                    para chegar nele a tempo, seja porque a partir dali o
                    destino fica inalcançável dentro de b_destino;
      - EM PARES  : i na posição k junto com j na posição l é impossível.
                    Para isso a DP é reiniciada a partir dos estados em que
                    i é o k-ésimo, o que capta violações que só aparecem
                    alguns passos adiante — inclusive no destino, que é
                    exatamente o caso que a versão frouxa deixava passar.

    Nota metodológica: esta é uma etapa de PRÉ-PROCESSAMENTO CLÁSSICO da
    formulação, no espírito da eliminação de arcos inviáveis usada na
    literatura de TSPTW. Ela determina o que é viável; a escolha entre as
    ordens viáveis, por distância, continua sendo do QAOA.

    Devolve (e, proibido_linear, proibido_par, partida)."""
    A, B = instancia["A"], instancia["B"]
    servico, C = instancia["servico"], instancia["C"]

    origem, destino = sub_rota[0], sub_rota[-1]
    livres = sub_rota[1:-1]
    K = len(livres)
    INF = float("inf")

    t0 = float(A[origem]) if t_inicial is None else float(t_inicial)
    partida = max(t0, float(A[origem])) + float(servico[origem])

    if K == 0:
        return {}, set(), set(), partida

    cheio = (1 << K) - 1

    def avanca(atual, i, j):
        return max(
            float(A[livres[j]]),
            atual + float(servico[livres[i]]) + float(C[livres[i], livres[j]]),
        )

    # --- D: chegada mais cedo, exata -------------------------------------
    D = [[INF] * K for _ in range(1 << K)]
    for i, cli in enumerate(livres):
        D[1 << i][i] = max(float(A[cli]), partida + float(C[origem, cli]))
    for mask in range(1 << K):
        for i in range(K):
            if not (mask >> i) & 1 or D[mask][i] == INF:
                continue
            for j in range(K):
                if (mask >> j) & 1:
                    continue
                novo = avanca(D[mask][i], i, j)
                if novo < D[mask | (1 << j)][j]:
                    D[mask | (1 << j)][j] = novo

    # --- Comp: custo mínimo restante até o destino ------------------------
    Comp = [[INF] * K for _ in range(1 << K)]
    for i, cli in enumerate(livres):
        Comp[cheio][i] = float(servico[cli]) + float(C[cli, destino])
    for mask in range(cheio - 1, -1, -1):
        for i in range(K):
            if not (mask >> i) & 1:
                continue
            melhor = INF
            for j in range(K):
                if (mask >> j) & 1:
                    continue
                v = (
                    float(servico[livres[i]])
                    + float(C[livres[i], livres[j]])
                    + Comp[mask | (1 << j)][j]
                )
                melhor = min(melhor, v)
            Comp[mask][i] = melhor

    def mascaras(k):
        return [m for m in range(1 << K) if bin(m).count("1") == k]

    # --- proibições lineares ---------------------------------------------
    e = {}
    proibido_linear = set()
    for i, cli in enumerate(livres):
        for k in range(1, K + 1):
            cands = [m for m in mascaras(k) if (m >> i) & 1 and D[m][i] < INF]
            if not cands:
                e[(i, k)] = INF
                proibido_linear.add((i, k))
                continue
            e[(i, k)] = min(D[m][i] for m in cands)
            ate_destino = min(D[m][i] + Comp[m][i] for m in cands)
            if e[(i, k)] > B[cli] or ate_destino > B[destino]:
                proibido_linear.add((i, k))

    # --- proibições em pares ---------------------------------------------
    proibido_par = set()
    for i in range(K):
        for k in range(1, K + 1):
            if (i, k) in proibido_linear:
                continue
            # reinicia a DP a partir dos estados em que i é o k-ésimo
            Dik = [[INF] * K for _ in range(1 << K)]
            for m in mascaras(k):
                if (m >> i) & 1 and D[m][i] < INF:
                    Dik[m][i] = min(Dik[m][i], D[m][i])
            for mask in range(1 << K):
                for a in range(K):
                    if not (mask >> a) & 1 or Dik[mask][a] == INF:
                        continue
                    for b in range(K):
                        if (mask >> b) & 1:
                            continue
                        novo = avanca(Dik[mask][a], a, b)
                        if novo < Dik[mask | (1 << b)][b]:
                            Dik[mask | (1 << b)][b] = novo
            for j in range(K):
                if j == i:
                    continue
                for l in range(k + 1, K + 1):
                    cands = [
                        m for m in mascaras(l)
                        if (m >> j) & 1 and (m >> i) & 1 and Dik[m][j] < INF
                    ]
                    if not cands:
                        proibido_par.add((i, k, j, l))
                        continue
                    cedo = min(Dik[m][j] for m in cands)
                    ate_destino = min(Dik[m][j] + Comp[m][j] for m in cands)
                    if cedo > B[livres[j]] or ate_destino > B[destino]:
                        proibido_par.add((i, k, j, l))

    return e, proibido_linear, proibido_par, partida


class FatiaInviavelError(RuntimeError):
    """Nenhuma ordem dos clientes da fatia respeita as janelas de tempo, dado
    o relógio real na origem. Não há o que o QAOA possa resolver."""

    def __init__(self, mensagem, fatia, t_origem):
        super().__init__(mensagem)
        self.fatia = list(fatia)
        self.t_origem = t_origem


def fatia_tem_permutacao_viavel(sub_rota, instancia, t_inicial=None):
    """PRÉ-PROCESSAMENTO clássico: existe alguma ordem dos clientes livres
    da fatia que respeite todas as janelas, inclusive a do destino fixo?

    Busca em profundidade com poda por janela (exata, e barata para as
    fatias pequenas do path-slicing). Não escolhe a ordem — só diz se existe
    uma; a escolha entre as viáveis continua sendo do QAOA."""
    A, B = instancia["A"], instancia["B"]
    servico, C = instancia["servico"], instancia["C"]
    origem, destino = sub_rota[0], sub_rota[-1]
    livres = list(sub_rota[1:-1])

    t0 = float(A[origem]) if t_inicial is None else float(t_inicial)
    partida = max(t0, float(A[origem])) + float(servico[origem])

    def chegada(t_saida, ant, prox):
        return max(t_saida + float(C[ant, prox]), float(A[prox]))

    def dfs(ant, t_saida, restantes):
        if not restantes:
            return chegada(t_saida, ant, destino) <= B[destino] + 1e-9
        for c in restantes:
            t = chegada(t_saida, ant, c)
            if t > B[c] + 1e-9:
                continue
            if dfs(c, t + float(servico[c]), restantes - {c}):
                return True
        return False

    return dfs(origem, partida, frozenset(livres))


def escala_penalidade(sub_rota, instancia, fator=2.0):
    """Escala base das penalidades, a partir da maior aresta da fatia."""
    nos = list(sub_rota)
    sub = instancia["C"][np.ix_(nos, nos)]
    return max(float(fator) * float(sub.max()), 1.0)


def soma_penal_linear(alpha):
    """Termo linear da expansão de (sum x - 1)^2: o coeficiente diagonal de
    cada x_i é -alpha (a constante é omitida)."""
    return -alpha


def construir_qubo_calibrado(sub_rota, instancia, fator_penalidade=1.3,
                             t_inicial=None, limite=LIMITE_FORCA_BRUTA,
                             tentativas=6):
    """Monta o QUBO e VERIFICA, por enumeração, que o seu mínimo global é
    uma permutação válida. Se não for, aumenta o fator de penalidade e
    refaz, até no máximo 'tentativas' vezes.

    Existe um equilíbrio real aqui. Penalidade alta demais garante a
    validade, mas achata o sinal útil: com fator 2.0 a diferença de energia
    entre a melhor permutação e a segunda cai para ~1% do total, e o QAOA
    não consegue resolver essa distinção. Com fator 1.3 o contraste sobe
    para ~2% e o ótimo continua sendo uma permutação. A calibração
    automática busca o menor fator que ainda seja seguro, em vez de fixar
    um valor conservador no escuro.

    Para fatias grandes demais para enumerar, mantém o fator pedido."""
    fator = float(fator_penalidade)
    for _ in range(tentativas):
        Q, n_vars, mapa, origem, pesos = construir_qubo(
            sub_rota, instancia, fator_penalidade=fator, t_inicial=t_inicial
        )
        if n_vars == 0 or n_vars > limite:
            pesos["fator"] = fator
            pesos["calibrado"] = False
            return Q, n_vars, mapa, origem, pesos

        h, Jm, offset = qubo_para_ising(Q, n_vars)
        energias = _todas_as_energias(n_vars, h, Jm, offset)
        bs = format(int(np.argmin(energias)), f"0{n_vars}b")
        if eh_permutacao_valida(bs, mapa, len(sub_rota) - 2):
            pesos["fator"] = fator
            pesos["calibrado"] = True
            return Q, n_vars, mapa, origem, pesos
        fator *= 1.5

    pesos["fator"] = fator
    pesos["calibrado"] = False
    return Q, n_vars, mapa, origem, pesos


def propagar_permitidos(K, permitidos):
    """Propagação de restrições sobre o conjunto de pares (cliente, posição)
    ainda possíveis: se um cliente só cabe em uma posição, nenhum outro pode
    ocupá-la; se uma posição só admite um cliente, esse cliente não pode ir
    para outra. Repete até estabilizar. PRÉ-PROCESSAMENTO clássico: só remove
    o que é comprovadamente impossível."""
    permitidos = set(permitidos)
    mudou = True
    while mudou:
        mudou = False
        for i in range(K):
            pos = [k for k in range(1, K + 1) if (i, k) in permitidos]
            if len(pos) == 1:
                k = pos[0]
                for j in range(K):
                    if j != i and (j, k) in permitidos:
                        permitidos.discard((j, k))
                        mudou = True
        for k in range(1, K + 1):
            cli = [i for i in range(K) if (i, k) in permitidos]
            if len(cli) == 1:
                i = cli[0]
                for l in range(1, K + 1):
                    if l != k and (i, l) in permitidos:
                        permitidos.discard((i, l))
                        mudou = True
    return permitidos


def construir_qubo(sub_rota, instancia, alpha=None, beta=None,
                   fator_penalidade=1.3, t_inicial=None):
    """Constrói H_Total = H_Custo + alpha*H_Fluxo + beta*H_Tempo para a
    fatia, com variáveis binárias x_{i,k} (cliente i na posição k).

    AMBOS os extremos da fatia são fixos: o primeiro nó é a origem e o
    último é o destino, otimizando-se apenas o interior. Fixar só a origem,
    como numa versão anterior, deixava o nó de emenda flutuar para dentro
    da fatia — e como esse nó é a origem da fatia SEGUINTE, o relógio e o
    ponto de partida do próximo QUBO ficavam calculados a partir de um nó
    que não era o último da rota. Fixar os dois extremos costura as fatias
    corretamente e ainda reduz o número de variáveis para (K-2) x (K-2).

    H_Tempo (versão temporal consistente, ainda sem variáveis de folga):
      (a) termo LINEAR: se nem no melhor cenário o cliente i consegue
          chegar à posição k dentro de b_i, penaliza x_{i,k} na diagonal.
          Usa espaço que já existe, sem nenhum qubit adicional, e é o que
          amarra tempo absoluto a posição;
      (b) termo QUADRÁTICO para todo par de posições k < l (e não só
          adjacentes, como antes): se colocar i em k inviabiliza j em l,
          penaliza o par. Entre posições vizinhas o teste usa o custo
          exato c_ij; entre posições distantes, o limitante inferior.
      As penalidades são BINÁRIAS: uma alocação impossível custa beta,
      independentemente do tamanho do atraso.

    A ordem de montagem importa: H_Custo e H_Tempo são construídos primeiro,
    e só então alpha é dimensionado como fator x (maior carga de
    coeficientes sobre uma única variável). Calibrar alpha apenas pela maior
    aresta, como antes, deixaria H_Fluxo ser dominado por um H_Tempo pesado
    e reintroduziria o problema de o mínimo do QUBO não ser uma permutação.

    't_inicial' é o tempo de chegada REAL na origem da fatia, propagado pelo
    pipeline. Sem ele, cada fatia é tratada como se o dia começasse do zero.

    Retorna (Q, n_vars, mapa, origem, pesos)."""
    C = instancia["C"]
    B = instancia["B"]
    servico = instancia["servico"]

    origem = sub_rota[0]
    destino = sub_rota[-1]
    livres = sub_rota[1:-1]
    K = len(livres)

    if beta is None:
        beta = escala_penalidade(sub_rota, instancia, fator_penalidade)

    # --- H_Tempo (parte 1): análise temporal e ELIMINAÇÃO de variáveis ------
    e, proibido_linear, proibido_par, partida = analise_temporal(
        sub_rota, instancia, t_inicial
    )
    # Cada x_{i,k} que comprovadamente não cabe é REMOVIDO do problema (em
    # vez de ficar com uma penalidade beta): o QUBO ganha menos qubits e a
    # faixa de energia deixa de ser dominada por penalidades, que escondiam o
    # sinal de distância. É pré-processamento clássico da formulação; a
    # escolha entre as ordens viáveis continua sendo do QAOA.
    permitidos = propagar_permitidos(
        K, {(i, k) for i in range(K) for k in range(1, K + 1)} - proibido_linear
    )

    var = {}
    for i in range(K):
        for k in range(1, K + 1):
            if (i, k) in permitidos:
                var[(i, k)] = len(var)

    def var_id(i, k):
        return var[(i, k)]

    Q = {}

    def soma(chave, valor):
        Q[chave] = Q.get(chave, 0.0) + valor

    # --- H_Custo -----------------------------------------------------------
    # trecho origem -> posição 1
    for i, cli in enumerate(livres):
        if (i, 1) in var:
            soma((var_id(i, 1), var_id(i, 1)), C[origem, cli])

    # trecho posição K -> destino (o extremo fixo do outro lado)
    for i, cli in enumerate(livres):
        if (i, K) in var:
            soma((var_id(i, K), var_id(i, K)), C[cli, destino])

    for k in range(1, K):
        for i, ci in enumerate(livres):
            for j, cj in enumerate(livres):
                if i == j or (i, k) not in var or (j, k + 1) not in var:
                    continue
                v1, v2 = var_id(i, k), var_id(j, k + 1)
                soma((min(v1, v2), max(v1, v2)), C[ci, cj])

    # --- H_Tempo (parte 2): pares proibidos entre variáveis que sobraram ----
    penal_lineares = len(proibido_linear)
    penal_pares = 0
    for (i, k, j, l) in proibido_par:
        if (i, k) not in var or (j, l) not in var:
            continue  # um dos lados já foi eliminado: o par perdeu o sentido
        v1, v2 = var_id(i, k), var_id(j, l)
        soma((min(v1, v2), max(v1, v2)), beta)
        penal_pares += 1

    # --- H_Fluxo: calibrado DEPOIS, para dominar custo + tempo -------------
    if alpha is None:
        carga = {}
        for (a, b), coef in Q.items():
            carga[a] = carga.get(a, 0.0) + abs(coef)
            if b != a:
                carga[b] = carga.get(b, 0.0) + abs(coef)
        alpha = max(float(fator_penalidade) * max(carga.values(), default=0.0), 1.0)

    # um cliente em exatamente uma posição / uma posição com exatamente um
    # cliente, só entre as variáveis que existem
    grupos = [
        [var_id(i, k) for k in range(1, K + 1) if (i, k) in var] for i in range(K)
    ] + [
        [var_id(i, k) for i in range(K) if (i, k) in var] for k in range(1, K + 1)
    ]
    for vs in grupos:
        for v1 in vs:
            soma((v1, v1), soma_penal_linear(alpha))
            for v2 in vs:
                if v1 != v2:
                    soma((min(v1, v2), max(v1, v2)), alpha)

    n_vars = len(var)
    mapa = {vid: (livres[i], k) for (i, k), vid in var.items()}
    pesos = dict(
        destino=destino,
        alpha=alpha,
        beta=beta,
        t_inicial=partida,
        penalidades_lineares=penal_lineares,
        penalidades_pares=penal_pares,
        variaveis_eliminadas=K * K - n_vars,
    )
    return Q, n_vars, mapa, origem, pesos


# ---------------------------------------------------------------------------
# QUBO -> Hamiltoniano de Custo (Pauli-Z) — Seção 2.4.1, Equação 8
# ---------------------------------------------------------------------------

def qubo_para_ising(Q, n_vars):
    """Converte o QUBO (x_i em {0,1}) nos coeficientes do Hamiltoniano de
    Ising equivalente (z_i em {-1,1}) via x_i = (1 - z_i)/2."""
    h = np.zeros(n_vars)
    Jm = np.zeros((n_vars, n_vars))
    offset = 0.0

    for (a, b), coef in Q.items():
        if a == b:
            offset += coef / 2
            h[a] += -coef / 2
        else:
            offset += coef / 4
            h[a] += -coef / 4
            h[b] += -coef / 4
            Jm[a, b] += coef / 4

    return h, Jm, offset


def energia_ising(bits, h, Jm, offset):
    z = 1 - 2 * np.asarray(bits)
    e = offset + np.dot(h, z)
    for a, b in np.argwhere(Jm != 0):
        e += Jm[a, b] * z[a] * z[b]
    return float(e)


def _todas_as_energias(n_vars, h, Jm, offset):
    """Enumeração vetorizada de todas as 2^n_vars energias. Usada só nas
    métricas de diagnóstico, para n_vars pequeno."""
    idx = np.arange(2 ** n_vars, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n_vars - 1, -1, -1)) & 1).astype(np.float64)
    z = 1.0 - 2.0 * bits
    e = offset + z @ h
    for a, b in np.argwhere(Jm != 0):
        e += Jm[a, b] * z[:, a] * z[:, b]
    return e


# ---------------------------------------------------------------------------
# Validação estrutural de uma bitstring
# ---------------------------------------------------------------------------

def eh_permutacao_valida(bitstring, mapa, K):
    """True se a bitstring codifica uma permutação legítima: exatamente um
    cliente por posição e exatamente uma posição por cliente."""
    bits = [int(c) for c in bitstring]
    posicoes, clientes = {}, {}
    for var_id, (cliente, k) in mapa.items():
        if bits[var_id]:
            posicoes[k] = posicoes.get(k, 0) + 1
            clientes[cliente] = clientes.get(cliente, 0) + 1
    return (
        len(posicoes) == K
        and len(clientes) == K
        and all(v == 1 for v in posicoes.values())
        and all(v == 1 for v in clientes.values())
    )


# ---------------------------------------------------------------------------
# 4.3.4 Execução Quântica: QAOA no Amazon Braket LocalSimulator
# ---------------------------------------------------------------------------

def montar_circuito_qaoa(n_vars, h, Jm, gamma, beta_ang, camadas):
    """Circuito QAOA de p camadas: superposição uniforme, alternando o
    operador de custo exp(-i*gamma*H_C) e o de mixing exp(-i*beta*sum X)."""
    _exigir_braket()
    circ = Circuit()
    for q in range(n_vars):
        circ.h(q)

    for p in range(camadas):
        g = gamma[p]
        for q in range(n_vars):
            if h[q] != 0:
                circ.rz(q, 2 * g * h[q])
        for a, b in np.argwhere(Jm != 0):
            if a < b:
                jab = Jm[a, b]
                circ.cnot(int(a), int(b))
                circ.rz(int(b), 2 * g * jab)
                circ.cnot(int(a), int(b))
        bta = beta_ang[p]
        for q in range(n_vars):
            circ.rx(q, 2 * bta)

    circ.probability()
    return circ


def energias_qubo(indices, n_vars, h, Jm, offset):
    """Energia do QUBO (na forma de Ising) de cada configuração, dada pelo
    índice inteiro da bitstring. Vetorizada."""
    indices = np.asarray(indices, dtype=np.int64)
    bits = ((indices[:, None] >> np.arange(n_vars - 1, -1, -1)) & 1).astype(float)
    z = 1.0 - 2.0 * bits
    e = offset + z @ h
    for a, b in np.argwhere(Jm != 0):
        e += Jm[a, b] * z[:, a] * z[:, b]
    return e


def normalizar_hamiltoniano(h, Jm):
    """NORMALIZAÇÃO DO HAMILTONIANO NO CIRCUITO. Os coeficientes do QUBO
    chegam a centenas ou milhares (as penalidades de permutação dominam),
    e o circuito aplica fases 2*gamma*h. Com gamma sorteado em [0, pi] a
    fase dá centenas de voltas, a paisagem do valor esperado vira ruído e o
    otimizador de ângulos não aprende nada: medido com a distribuição exata,
    P(permutação válida) ficava no nível do acaso (~1%) para qualquer número
    de camadas. Dividir h e J por um escalar positivo não muda o ótimo nem a
    ordem das energias; só coloca gamma na escala em que ele consegue
    trabalhar. O valor esperado continua sendo calculado com as energias
    originais.

    Os ângulos salvos em disco valem para o hamiltoniano NORMALIZADO, então
    a calibração e a medição precisam usar esta mesma função."""
    escala = max(float(np.abs(h).max(initial=0.0)),
                 float(np.abs(Jm).max(initial=0.0)), 1e-12)
    return h / escala, Jm / escala


def medir_qaoa(dispositivo, n_vars, h, Jm, offset, gamma, beta,
               amostras=500, topo=64):
    """Execução QUÂNTICA do QAOA com ângulos JÁ DEFINIDOS: monta o circuito,
    mede com 'amostras' shots no 'dispositivo' (LocalSimulator ou Braket) e
    devolve o ranking das bitstrings medidas. Não otimiza nada: de onde os
    ângulos vêm (calibração ou arquivo) é decidido fora deste módulo.

    SELEÇÃO POR ENERGIA (e não por frequência). As bitstrings medidas são
    ordenadas pela ENERGIA do QUBO, da menor para a maior. Isso é a prática
    padrão de uso do QAOA como amostrador: guarda-se a melhor amostra
    observada, não a moda da distribuição.

    O motivo é concreto: com 200 shots sobre 2^16 = 65536 estados, TODAS as
    200 bitstrings medidas empatam na mesma frequência (1/200), e o argmax
    vira um desempate arbitrário — ruído. Medido na instância rc_201.2, a
    seleção por frequência caía no rank 64323/65536 (pior que o acaso)
    enquanto a seleção por energia, com as mesmas 200 medidas, dava rank
    554/65536.

    A medição é sempre amostrada, nunca exaustiva: a seleção não pode varrer
    o espaço inteiro e "achar" o ótimo por força bruta disfarçada.

    Devolve (ranking, tempo), com ranking = [(bitstring, prob, energia)]
    ordenado por energia crescente e truncado em 'topo'."""
    _exigir_braket()
    h_circ, Jm_circ = normalizar_hamiltoniano(h, Jm)
    circ = montar_circuito_qaoa(
        n_vars, h_circ, Jm_circ, gamma, beta, len(gamma)
    )

    t0 = time_module.time()
    probs = dispositivo.probabilidades(circ, amostras)
    tempo = time_module.time() - t0

    medidos = np.flatnonzero(probs)
    if medidos.size == 0:
        medidos = np.array([int(np.argmax(probs))])

    energias = energias_qubo(medidos, n_vars, h, Jm, offset)
    ordem = np.argsort(energias)[: min(topo, medidos.size)]
    ranking = [
        (
            format(int(medidos[i]), f"0{n_vars}b"),
            float(probs[medidos[i]]),
            float(energias[i]),
        )
        for i in ordem
    ]

    return ranking, tempo


# ---------------------------------------------------------------------------
# Métricas de qualidade da etapa quântica
# ---------------------------------------------------------------------------

def metricas_qaoa(n_vars, h, Jm, offset, bitstring, mapa, K,
                  limite=LIMITE_FORCA_BRUTA):
    """Compara a solução do QAOA com o ÓTIMO EXATO do QUBO, obtido por
    enumeração das 2^n_vars configurações (só para fatias pequenas).

    Devolve dict com:
      - energia_qaoa, energia_otima, energia_pior
      - razao_aproximacao : (pior - qaoa) / (pior - otimo), em [0, 1];
                            1.0 significa que o QAOA achou o ótimo
      - rank              : posição da solução do QAOA no espectro (1 = ótimo)
      - percentil         : fração de estados piores que o encontrado
      - otimo_valido      : se o ótimo do QUBO é uma permutação válida —
                            um NÃO aqui indica penalidade subdimensionada,
                            isto é, formulação mal calibrada
      - fracao_validas    : fração do espaço que são permutações válidas

    Retorna None se n_vars exceder o limite de enumeração."""
    if n_vars > limite:
        return None

    energias = _todas_as_energias(n_vars, h, Jm, offset)
    e_qaoa = energia_ising([int(c) for c in bitstring], h, Jm, offset)
    e_min, e_max = float(energias.min()), float(energias.max())

    idx_otimo = int(np.argmin(energias))
    bs_otima = format(idx_otimo, f"0{n_vars}b")

    rank = int(np.sum(energias < e_qaoa - 1e-9)) + 1
    total = int(energias.size)

    if n_vars <= 16:
        validas = sum(
            1 for i in range(total)
            if eh_permutacao_valida(format(i, f"0{n_vars}b"), mapa, K)
        )
        fracao_validas = validas / total
    else:
        fracao_validas = None

    denom = e_max - e_min
    razao = 1.0 if denom <= 1e-12 else (e_max - e_qaoa) / denom

    return dict(
        energia_qaoa=e_qaoa,
        energia_otima=e_min,
        energia_pior=e_max,
        razao_aproximacao=float(razao),
        rank=rank,
        total_estados=total,
        percentil=1.0 - (rank - 1) / total,
        bitstring_otima=bs_otima,
        otimo_valido=eh_permutacao_valida(bs_otima, mapa, K),
        fracao_validas=fracao_validas,
    )


# ---------------------------------------------------------------------------
# 4.3.5 Pós-processamento: decodificação + busca local
# ---------------------------------------------------------------------------

def decodificar_bitstring(ranking, mapa, livres, K, origem, instancia):
    """Decodifica a saída do QAOA em uma sequência de visitas.

    'ranking' é a lista [(bitstring, probabilidade), ...] em ordem
    decrescente de probabilidade (uma string solta também é aceita).

    Estratégia, nesta ordem:
      1. usa a bitstring MAIS PROVÁVEL que seja uma permutação válida —
         essa é a decodificação fiel, em que a resposta quântica define
         integralmente a ordem de visita;
      2. se nenhuma medida for válida, cai na mais provável e completa as
         lacunas gulosamente (menor tempo de chegada entre os clientes
         ainda não usados), CONTABILIZANDO quantas posições precisaram de
         reparo.

    Devolve (sequencia, info), com info registrando se a decodificação foi
    fiel ou reparada — métrica essencial para saber se a etapa quântica
    está de fato influenciando o resultado."""
    if isinstance(ranking, str):
        ranking = [(ranking, 1.0, float("nan"))]
    ranking = [tuple(item) + (float("nan"),) * (3 - len(item)) for item in ranking]

    escolhida, prob, posicao, valida = ranking[0][0], ranking[0][1], 1, False
    for pos, (bs, p, _e) in enumerate(ranking, start=1):
        if eh_permutacao_valida(bs, mapa, K):
            escolhida, prob, posicao, valida = bs, p, pos, True
            break

    bits = [int(c) for c in escolhida]
    atribuicoes = {}
    for var_id, (cliente, k) in mapa.items():
        if bits[var_id]:
            atribuicoes.setdefault(k, []).append(cliente)

    A, servico, C = instancia["A"], instancia["servico"], instancia["C"]
    usados = set()
    sequencia = []
    no_anterior = origem
    t_atual = float(A[origem])
    reparos = 0      # posições sem nenhum candidato: preenchidas do zero
    ambiguas = 0     # posições com mais de um candidato: desempate guloso

    for k in range(1, K + 1):
        candidatos = [c for c in atribuicoes.get(k, []) if c not in usados]
        if not candidatos:
            candidatos = [c for c in livres if c not in usados]
            reparos += 1
        elif len(candidatos) > 1:
            ambiguas += 1

        def chegada(c, _ant=no_anterior, _t=t_atual):
            return max(_t + servico[_ant] + C[_ant, c], A[c])

        escolhido = min(candidatos, key=chegada)
        sequencia.append(escolhido)
        usados.add(escolhido)
        t_atual = chegada(escolhido)
        no_anterior = escolhido

    info = dict(
        bitstring=escolhida,
        probabilidade=prob,
        valida=valida,
        posicao_no_ranking=posicao,
        reparos=reparos,
        ambiguas=ambiguas,
        decididas_pelo_qaoa=K - reparos - ambiguas,
        posicoes=K,
    )
    return sequencia, info


def busca_local_restrita(sub_rota_quantica, origem, instancia):
    """Refino da fatia devolvida pelo QAOA, com a origem fixa e o fim
    livre (a fatia é um caminho aberto, não um ciclo).

    Usa o VND e o critério lexicográfico (violação, distância) do módulo
    clássico: reduzir violação de janela tem prioridade sobre encurtar a
    distância."""
    seq = [origem] + list(sub_rota_quantica)
    melhor, valor = vnd(seq, instancia, fixo_fim=False)
    return melhor[1:], valor  # remove a origem fixa


def busca_local_global(rota_completa, instancia):
    """Reconciliação final sobre a rota global (todas as fatias
    concatenadas), com o ciclo fechado no depósito e o relógio real
    acumulado desde a saída.

    ATENÇÃO METODOLÓGICA: esta etapa usa o MESMO VND do algoritmo
    clássico. Ela costuma dominar o resultado final, de modo que a rota
    entregue aqui pode coincidir com a do baseline mesmo quando as fatias
    quânticas diferiram. Por isso resolver_quantico() também registra a
    rota ANTES desta etapa (chave 'rota_pre_refino'), permitindo medir a
    contribuição isolada do QAOA."""
    deposito = rota_completa[0]
    seq = list(rota_completa) + [deposito]
    melhor, valor = vnd(seq, instancia, fixo_fim=True)
    return melhor[:-1], valor  # remove o depósito duplicado no fim


# ---------------------------------------------------------------------------
# Pipeline quântico (Quadro 6 do TCC)
# ---------------------------------------------------------------------------

def resolver_quantico(instancia, provedor_angulos, dispositivo=None, K=4,
                      amostras=2000, fator_penalidade=1.3, metricas=True,
                      busca_local="nenhuma", construcao="auto", verbose=True):
    """Executa o pipeline de path-slicing sobre uma instância JÁ CARREGADA
    (o carregamento fica em main.py).

    'provedor_angulos' é um callable que devolve os ângulos (gamma, beta) do
    QAOA de cada fatia: ver qaoa_calibracao.py. Este módulo não otimiza
    ângulo nenhum; só monta o circuito com os que recebe e mede. Assinatura:
        provedor_angulos(Q, n_vars, h, Jm, offset, contexto)
            -> (gamma, beta, tempo_quantico)
    'dispositivo' é onde a medição roda (padrão: LocalSimulator).

    'busca_local' define quanta busca local CLÁSSICA entra no pipeline:
      - "nenhuma"  : nenhuma. A rota é o que o solver produziu, e só.
                     É o modo correto para comparar o método quântico
                     contra uma heurística clássica sem contaminação.
      - "fatia"    : VND dentro de cada fatia (busca_local_restrita).
      - "completa" : VND por fatia + VND global sobre a rota inteira.
                     Nesse modo o pipeline termina com o mesmo operador do
                     baseline clássico e os dois convergem para o mesmo
                     ótimo local — ver a nota em busca_local_global.

    Devolve um dicionário no mesmo formato de resolver_classico()."""
    if busca_local not in ("nenhuma", "fatia", "completa"):
        raise ValueError(f"busca_local inválida: {busca_local!r}")
    _exigir_braket()
    if dispositivo is None:
        dispositivo = criar_dispositivo(aws=False)
    base_dispositivo = dispositivo.instantaneo()

    if verbose:
        print("=" * 70)
        print(
            f"ALGORITMO QUÂNTICO — Path-Slicing + QUBO + QAOA "
            f"(busca local: {busca_local})"
        )
        print("=" * 70)
        print(
            f"[Q1] Instância: {instancia['n']} nós "
            f"(depósito + {instancia['n'] - 1} clientes)"
        )

    # 1) Rota inicial (heurística) + Path-Slicing
    rota_inicial, nome_construcao = construir_rota_inicial(instancia, construcao, K)
    deposito = rota_inicial[0]
    # O depósito entra como ÚLTIMO NÓ da rota antes de fatiar. Assim todas as
    # fatias têm os dois extremos fixos e o mesmo tamanho (K-2 clientes
    # livres), inclusive a final. Fatiar a rota aberta e anexar o depósito
    # depois dava à última fatia um cliente livre a mais: 16 variáveis contra
    # 9 nas demais, e era quase só ali que o QAOA errava o ótimo.
    fatias = path_slicing(list(rota_inicial) + [deposito], K)
    if verbose:
        print(f"[Q2] Rota inicial ({nome_construcao}): {rota_inicial}")
        print(f"     Path-slicing (K={K}): {fatias}")

    rota_global = [rota_inicial[0]]
    rota_crua = [rota_inicial[0]]  # só QAOA + decodificação, sem busca local
    tempo_classico_total = 0.0
    tempo_quantico_total = 0.0
    diagnostico = []

    for idx_fatia, fatia in enumerate(fatias, start=1):
        ultima = idx_fatia == len(fatias)
        # a fatia tem os DOIS extremos fixos; na última, o destino é o
        # depósito, onde o ciclo fecha.
        fatia_ext = list(fatia)
        if verbose:
            print(
                f"\n--- Sub-rota {idx_fatia}/{len(fatias)}: {fatia_ext} "
                f"(origem {fatia_ext[0]}, destino {fatia_ext[-1]} fixos) ---"
            )
        t0 = time_module.time()

        # 2) Tradutor QUBO. O relógio real na origem da fatia vem da rota
        # já construída, e não de a_origem: é o que permite ao H_Tempo saber
        # que horas são quando a fatia começa.
        t_origem = tempos_de_chegada(rota_global, instancia)[-1]
        if not fatia_tem_permutacao_viavel(fatia_ext, instancia, t_origem):
            prazos = ", ".join(
                f"{c}(b={instancia['B'][c]:.0f})" for c in fatia_ext[1:-1]
            )
            raise FatiaInviavelError(
                f"Sub-rota {idx_fatia}/{len(fatias)} {fatia_ext} é inviável: "
                f"partindo de {fatia_ext[0]} em t={t_origem:.1f}, nenhuma ordem "
                f"dos clientes [{prazos}] cumpre as janelas (destino "
                f"{fatia_ext[-1]}, b={instancia['B'][fatia_ext[-1]]:.0f}). "
                f"O problema está na rota inicial/composição das fatias, não "
                f"no QAOA.",
                fatia_ext, t_origem,
            )
        Q, n_vars, mapa, origem, pesos = construir_qubo_calibrado(
            fatia_ext, instancia, fator_penalidade=fator_penalidade,
            t_inicial=t_origem,
        )
        if n_vars == 0:  # fatia sem interior: nada a otimizar
            tempo_classico_total += time_module.time() - t0
            if not ultima:
                rota_global.append(fatia_ext[-1])
                rota_crua.append(fatia_ext[-1])
            continue
        h, Jm, offset = qubo_para_ising(Q, n_vars)
        tempo_classico_total += time_module.time() - t0
        if verbose:
            print(
                f"    QUBO: {n_vars} variáveis | chegada na origem "
                f"t={t_origem:.1f} | alpha={pesos['alpha']:.1f} "
                f"(fator {pesos['fator']:.2f}"
                + (", validado)" if pesos.get("calibrado") else ")")
            )
            print(
                f"    H_Tempo: {pesos['penalidades_lineares']} alocações "
                f"impossíveis, {pesos['penalidades_pares']} pares proibidos"
            )

        # 3) Execução quântica: ângulos (calibrados ou lidos do arquivo) +
        # medição do circuito
        gamma, beta, t_ang = provedor_angulos(
            Q, n_vars, h, Jm, offset,
            dict(fatia=fatia_ext, t_origem=t_origem),
        )
        ranking, t_med = medir_qaoa(
            dispositivo, n_vars, h, Jm, offset, gamma, beta, amostras=amostras
        )
        tempo_quantico_total += t_ang + t_med

        # 4) Pós-processamento
        t0 = time_module.time()
        livres = fatia_ext[1:-1]
        sub_rota_quantica, info = decodificar_bitstring(
            ranking, mapa, livres, len(livres), origem, instancia
        )

        met = None
        if metricas:
            met = metricas_qaoa(
                n_vars, h, Jm, offset, info["bitstring"], mapa, len(livres)
            )

        if busca_local in ("fatia", "completa"):
            sub_rota_refinada, valor_fatia = busca_local_restrita(
                sub_rota_quantica, origem, instancia
            )
        else:
            sub_rota_refinada = list(sub_rota_quantica)
            valor_fatia = avaliar([origem] + sub_rota_refinada, instancia)
        tempo_classico_total += time_module.time() - t0

        if verbose:
            estado = (
                f"permutação válida (amostra {info['posicao_no_ranking']} por energia)"
                if info["valida"]
                else (
                    f"inválida — {info['decididas_pelo_qaoa']}/{info['posicoes']} "
                    f"posições vieram do QAOA, {info['ambiguas']} ambíguas, "
                    f"{info['reparos']} reparadas"
                )
            )
            print(f"    Bitstring escolhida: {info['bitstring']}  [{estado}]")
            if met:
                print(
                    f"    QAOA vs ótimo do QUBO: razão de aproximação "
                    f"{met['razao_aproximacao']:.3f} | rank {met['rank']}/"
                    f"{met['total_estados']}"
                )
                if not met["otimo_valido"]:
                    print(
                        "    [!] O ótimo do QUBO NÃO é permutação válida — "
                        "aumente --penalidade."
                    )
            print(f"    Decodificada: {sub_rota_quantica}")
            if busca_local in ("fatia", "completa"):
                print(f"    Sub-rota refinada (VND): {sub_rota_refinada}")

        diagnostico.append(
            dict(
                fatia=fatia_ext,
                n_vars=n_vars,
                alpha=pesos["alpha"],
                pesos=pesos,
                t_origem=t_origem,
                decodificacao=info,
                metricas=met,
                sub_rota_decodificada=sub_rota_quantica,
                sub_rota_refinada=sub_rota_refinada,
                valor_fatia=valor_fatia,
                angulos=dict(gamma=list(gamma), beta=list(beta)),
            )
        )
        rota_global.extend(sub_rota_refinada)
        rota_crua.extend(sub_rota_quantica)
        # o destino é o nó de emenda: entra agora na rota e vira a origem
        # da próxima fatia. Na última, o destino é o depósito e não entra.
        if not ultima:
            rota_global.append(fatia_ext[-1])
            rota_crua.append(fatia_ext[-1])

    valor_cru = avaliar(rota_crua + [rota_crua[0]], instancia)

    # rota produzida pelas fatias, ANTES do refino global
    rota_pre_refino = list(rota_global)
    valor_pre_refino = avaliar(rota_pre_refino + [rota_pre_refino[0]], instancia)

    # 5) Reconciliação global (opcional)
    # Desligar esta etapa isola a contribuição da parte quântica: com ela
    # ligada, o pipeline termina com exatamente o mesmo VND que o baseline
    # clássico usa, partindo da mesma rota inicial — e os dois convergem
    # para o mesmo ótimo local em qualquer instância.
    if busca_local == "completa":
        t0 = time_module.time()
        rota_global, valor_global = busca_local_global(rota_global, instancia)
        tempo_classico_total += time_module.time() - t0
    else:
        valor_global = valor_pre_refino

    # 6) Execução do percurso real
    percurso = simular_percurso(rota_global, instancia, fechar_no_deposito=True)
    custo = custo_rota(list(rota_global) + [rota_global[0]], instancia["C"])
    tte = tempo_classico_total + tempo_quantico_total

    n_fatias = max(len(diagnostico), 1)
    validas = sum(1 for d in diagnostico if d["decodificacao"]["valida"])
    posicoes_tot = sum(d["decodificacao"]["posicoes"] for d in diagnostico) or 1
    posicoes_qaoa = sum(d["decodificacao"]["decididas_pelo_qaoa"] for d in diagnostico)
    razoes = [
        d["metricas"]["razao_aproximacao"] for d in diagnostico if d["metricas"]
    ]
    otimos_invalidos = sum(
        1 for d in diagnostico if d["metricas"] and not d["metricas"]["otimo_valido"]
    )

    return dict(
        algoritmo="quantico",
        busca_local=busca_local,
        rota=rota_global,
        custo=custo,
        violacao=valor_global[0],
        viavel=not percurso["inviavel"],
        percurso=percurso,
        rota_inicial=rota_inicial,
        valor_inicial=avaliar(list(rota_inicial) + [rota_inicial[0]], instancia),
        rota_crua=rota_crua,
        valor_cru=valor_cru,
        rota_pre_refino=rota_pre_refino,
        valor_pre_refino=valor_pre_refino,
        fatias=fatias,
        diagnostico=diagnostico,
        resumo_quantico=dict(
            fatias=n_fatias,
            decodificacoes_validas=validas,
            taxa_validas=validas / n_fatias,
            posicoes_decididas_pelo_qaoa=posicoes_qaoa,
            posicoes_totais=posicoes_tot,
            taxa_posicoes_qaoa=posicoes_qaoa / posicoes_tot,
            razao_aproximacao_media=(float(np.mean(razoes)) if razoes else None),
            otimos_qubo_invalidos=otimos_invalidos,
        ),
        # tempos medidos pelo próprio dispositivo (metadados das tarefas),
        # distintos do tsq, que é tempo de relógio e inclui rede e fila
        dispositivo=dict(
            nome=dispositivo.nome, **dispositivo.desde(base_dispositivo)
        ),
        tcl=tempo_classico_total,
        tsq=tempo_quantico_total,
        tte=tte,
    )