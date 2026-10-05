"""
classic_algorithm.py
===========================================================================
Resolução clássica do TSPTW (Traveling Salesman Problem with Time Windows)
por GVNS (General Variable Neighborhood Search), conforme a linha de
Da Silva & Urrutia (2010), "A General VNS heuristic for the traveling
salesman problem with time windows", Discrete Optimization 7(4).

Este módulo concentra TRÊS responsabilidades:

  (1) a FUNÇÃO OBJETIVO e as primitivas de avaliação temporal do TSPTW
      (custo, tempos de chegada, violação de janelas, simulação do
      percurso real). Elas ficam aqui porque são puramente clássicas, e
      o módulo quântico as importa para usar exatamente o mesmo critério;

  (2) as VIZINHANÇAS e o VND (Variable Neighborhood Descent), também
      reutilizados pelo pós-processamento do pipeline quântico;

  (3) o GVNS propriamente dito (shaking + VND + critério de aceitação),
      que é o baseline clássico de comparação.

---------------------------------------------------------------------------
CRITÉRIO LEXICOGRÁFICO
---------------------------------------------------------------------------
A avaliação de uma solução devolve o par ordenado

    f(s) = ( violacao_total(s) , distancia_total(s) )

e a comparação entre soluções é a comparação natural de tuplas. Ou seja:
primeiro minimiza-se a violação acumulada das janelas de tempo; só entre
soluções de mesma violação (em particular, entre as VIÁVEIS, com violação
zero) é que a distância decide. É o critério padrão da literatura de TSPTW
e evita o comportamento de aceitar por distância uma rota que estoura
janelas.

Note que violacao_total propaga o relógio ATÉ O FIM da rota mesmo depois de
uma janela estourada (não interrompe no primeiro erro). Isso é proposital:
dá à busca local um gradiente contínuo para seguir, permitindo sair de uma
região inviável. A interrupção "de verdade" acontece só em
simular_percurso(), que é a execução realista usada no relatório final.
"""

import random
import time as time_module

import numpy as np


# ---------------------------------------------------------------------------
# Primitivas de avaliação
# ---------------------------------------------------------------------------

def custo_rota(seq, C):
    """Distância (tempo de viagem puro) total percorrida na sequência."""
    return float(sum(C[seq[i], seq[i + 1]] for i in range(len(seq) - 1)))


def tempos_de_chegada(seq, instancia, t_inicial=None):
    """Tempos de chegada em cada nó da sequência, com espera permitida
    (chega antes de a_i => espera até a_i) e SEM interromper em violação.

        chegada_j = max( t_i + servico_i + c_ij , a_j )
    """
    A, servico, C = instancia["A"], instancia["servico"], instancia["C"]
    t0 = float(A[seq[0]]) if t_inicial is None else float(t_inicial)
    tempos = [max(t0, float(A[seq[0]]))]
    for i in range(1, len(seq)):
        anterior, atual = seq[i - 1], seq[i]
        chegada = tempos[-1] + servico[anterior] + C[anterior, atual]
        tempos.append(float(max(chegada, A[atual])))
    return tempos


def violacao_total(seq, instancia, t_inicial=None):
    """Soma dos atrasos max(0, chegada_i - b_i) ao longo de toda a
    sequência. Zero <=> solução viável."""
    B = instancia["B"]
    tempos = tempos_de_chegada(seq, instancia, t_inicial)
    return float(sum(max(0.0, t - B[no]) for no, t in zip(seq, tempos)))


def avaliar(seq, instancia, t_inicial=None):
    """Função objetivo lexicográfica: (violação, distância).

    Comparar duas soluções é comparar as tuplas devolvidas por esta
    função — menor é melhor."""
    return (
        violacao_total(seq, instancia, t_inicial),
        custo_rota(seq, instancia["C"]),
    )


def eh_viavel(seq, instancia, t_inicial=None, tol=1e-6):
    return violacao_total(seq, instancia, t_inicial) <= tol


def simular_percurso(rota, instancia, fechar_no_deposito=False):
    """Execução REALISTA da rota, nó a nó, com o relógio real: na primeira
    janela estourada o veículo para e o restante fica sem visitar.

    Diferente de violacao_total (que segue até o fim para orientar a
    busca), aqui o objetivo é relatar o que aconteceria de fato.

    Retorna dict com: percorrido, nao_visitados, inviavel, tempos,
    no_falha, chegada_falha, retorno_deposito."""
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
            nao_visitados = [inicio]

    return dict(
        percorrido=percorrido,
        nao_visitados=nao_visitados,
        inviavel=inviavel,
        tempos=tempos,
        no_falha=no_falha,
        chegada_falha=chegada_falha,
        retorno_deposito=retorno_deposito,
    )


def viola_janela(seq, instancia):
    """True se a sequência não pode ser executada inteira dentro das
    janelas (usa a simulação do percurso real)."""
    return simular_percurso(seq, instancia)["inviavel"]


# ---------------------------------------------------------------------------
# Heurística construtiva
# ---------------------------------------------------------------------------

def rota_vizinho_mais_proximo(instancia):
    """Vizinho mais próximo sensível a janelas: a cada passo escolhe o
    cliente não visitado com menor tempo de chegada, priorizando os que
    ainda cabem na janela (atraso zero). A rota sempre contém todos os
    clientes; a inviabilidade é tratada depois, pela busca."""
    A, B, servico, C = (
        instancia["A"], instancia["B"], instancia["servico"], instancia["C"]
    )
    n = instancia["n"]

    deposito = 0
    nao_visitados = set(range(1, n))
    rota = [deposito]
    t_atual = float(A[deposito])
    atual = deposito

    while nao_visitados:
        candidatos = []
        for c in nao_visitados:
            chegada = max(t_atual + servico[atual] + C[atual, c], A[c])
            atraso = max(0.0, chegada - B[c])
            candidatos.append((atraso, chegada, c))
        atraso, chegada, escolhido = min(candidatos, key=lambda x: (x[0], x[1]))
        rota.append(escolhido)
        nao_visitados.remove(escolhido)
        t_atual, atual = chegada, escolhido

    return rota


def rota_insercao_mais_barata(instancia, semente=None):
    """Construção alternativa (usada nos reinícios do GVNS): inserção do
    cliente na posição que menos degrada a função objetivo lexicográfica.
    Com 'semente' embaralha a ordem de consideração dos clientes, gerando
    soluções iniciais diversas."""
    n = instancia["n"]
    pendentes = list(range(1, n))
    if semente is not None:
        random.Random(semente).shuffle(pendentes)

    rota = [0, 0]  # ciclo vazio: depósito -> depósito
    for cliente in pendentes:
        melhor_pos, melhor_val = 1, None
        for pos in range(1, len(rota)):
            candidato = rota[:pos] + [cliente] + rota[pos:]
            val = avaliar(candidato, instancia)
            if melhor_val is None or val < melhor_val:
                melhor_pos, melhor_val = pos, val
        rota = rota[:melhor_pos] + [cliente] + rota[melhor_pos:]

    return rota[:-1]  # devolve aberta (sem o depósito repetido no fim)


# ---------------------------------------------------------------------------
# Vizinhanças e VND
# ---------------------------------------------------------------------------
# Convenção: 'seq' é uma sequência qualquer cujo PRIMEIRO elemento é fixo
# (depósito, ou o nó de origem de uma fatia). Se fixo_fim=True, o ÚLTIMO
# elemento também é fixo (rota fechada no depósito). Os movimentos só
# tocam o miolo.

def _limite(seq, fixo_fim):
    return len(seq) - 1 if fixo_fim else len(seq)


def vizinhos_2opt(seq, fixo_fim=True):
    """Inversão de segmento: clássico 2-opt."""
    fim = _limite(seq, fixo_fim)
    for i in range(1, fim):
        for j in range(i + 1, fim):
            yield seq[:i] + seq[i : j + 1][::-1] + seq[j + 1 :]


def vizinhos_or_opt(seq, fixo_fim=True, tamanhos=(1, 2, 3)):
    """Or-opt: realocação de um segmento curto (1 a 3 clientes) para
    outra posição, preservando a orientação."""
    fim = _limite(seq, fixo_fim)
    for L in tamanhos:
        for i in range(1, fim - L + 1):
            segmento = seq[i : i + L]
            resto = seq[:i] + seq[i + L :]
            fim_resto = _limite(resto, fixo_fim)
            for pos in range(1, fim_resto + 1):
                if pos == i:
                    continue
                yield resto[:pos] + segmento + resto[pos:]


def vizinhos_swap(seq, fixo_fim=True):
    """Troca de dois clientes de posição (1-interchange)."""
    fim = _limite(seq, fixo_fim)
    for i in range(1, fim):
        for j in range(i + 1, fim):
            novo = list(seq)
            novo[i], novo[j] = novo[j], novo[i]
            yield novo


VIZINHANCAS = (vizinhos_or_opt, vizinhos_2opt, vizinhos_swap)


def vnd(seq, instancia, fixo_fim=True, t_inicial=None, vizinhancas=VIZINHANCAS):
    """Variable Neighborhood Descent com melhor-melhora dentro de cada
    vizinhança e reinício na primeira vizinhança a cada melhora.

    O critério de aceitação é SEMPRE lexicográfico (violação, distância):
    um movimento que reduz a violação é aceito mesmo que aumente a
    distância; entre movimentos de mesma violação, vence o mais curto."""
    melhor = list(seq)
    melhor_val = avaliar(melhor, instancia, t_inicial)

    l = 0
    while l < len(vizinhancas):
        cand_melhor, cand_val = None, melhor_val
        for candidato in vizinhancas[l](melhor, fixo_fim):
            val = avaliar(candidato, instancia, t_inicial)
            if val < cand_val:
                cand_melhor, cand_val = candidato, val
        if cand_melhor is not None:
            melhor, melhor_val = cand_melhor, cand_val
            l = 0  # melhorou: volta para a primeira vizinhança
        else:
            l += 1

    return melhor, melhor_val


# ---------------------------------------------------------------------------
# Shaking (perturbação)
# ---------------------------------------------------------------------------

def _shake_double_bridge(seq, rng, fixo_fim=True):
    """Perturbação double-bridge (4-opt): o movimento clássico de VNS/ILS
    para TSP, por não ser desfeito por um único passo de 2-opt."""
    fim = _limite(seq, fixo_fim)
    miolo = seq[1:fim]
    if len(miolo) < 4:
        return list(seq)
    p = sorted(rng.sample(range(1, len(miolo)), 3))
    a, b, c = p
    novo_miolo = miolo[:a] + miolo[c:] + miolo[b:c] + miolo[a:b]
    return seq[:1] + novo_miolo + seq[fim:]


def _shake_movimento_aleatorio(seq, rng, fixo_fim=True):
    """Um or-opt ou um swap aleatório."""
    fim = _limite(seq, fixo_fim)
    if fim - 1 < 2:
        return list(seq)
    novo = list(seq)
    if rng.random() < 0.5:
        i, j = rng.sample(range(1, fim), 2)
        novo[i], novo[j] = novo[j], novo[i]
    else:
        i = rng.randrange(1, fim)
        cliente = novo.pop(i)
        pos = rng.randrange(1, _limite(novo, fixo_fim) + 1)
        novo.insert(pos, cliente)
    return novo


def shake(seq, k, rng, fixo_fim=True):
    """Aplica k perturbações encadeadas. k cresce quando o GVNS não
    consegue melhorar, aumentando progressivamente o raio de busca."""
    novo = list(seq)
    for _ in range(k):
        if rng.random() < 0.5:
            novo = _shake_double_bridge(novo, rng, fixo_fim)
        else:
            novo = _shake_movimento_aleatorio(novo, rng, fixo_fim)
    return novo


# ---------------------------------------------------------------------------
# GVNS
# ---------------------------------------------------------------------------

def gvns(instancia, iteracoes=200, k_max=4, seed=42, verbose=True,
         reinicios_sem_melhora=60):
    """General Variable Neighborhood Search para o TSPTW.

    Estrutura (Da Silva & Urrutia, 2010, simplificada):

        s  <- construção gulosa + VND
        repete:
            s' <- shake(s, k)          # perturbação de intensidade k
            s''<- VND(s')              # descida em múltiplas vizinhanças
            se f(s'') < f(s):  s <- s'' ; k <- 1
            senão:             k <- k+1  (volta a 1 ao passar de k_max)

    com f lexicográfico (violação, distância). Após muitas iterações sem
    melhora, faz um reinício a partir de uma construção por inserção
    aleatorizada, mantendo sempre a melhor solução global encontrada.

    Devolve (rota_aberta, valor, historico) — rota_aberta começa no
    depósito e NÃO repete o depósito no fim."""
    rng = random.Random(seed)

    inicial = rota_vizinho_mais_proximo(instancia)
    atual = list(inicial) + [inicial[0]]  # fecha o ciclo
    atual, val_atual = vnd(atual, instancia, fixo_fim=True)

    melhor, melhor_val = list(atual), val_atual
    historico = [(0, melhor_val)]
    sem_melhora = 0

    for it in range(1, iteracoes + 1):
        k = 1
        while k <= k_max:
            candidato = shake(atual, k, rng, fixo_fim=True)
            candidato, val_cand = vnd(candidato, instancia, fixo_fim=True)

            if val_cand < val_atual:
                atual, val_atual = candidato, val_cand
                k = 1
                break
            k += 1

        if val_atual < melhor_val:
            melhor, melhor_val = list(atual), val_atual
            historico.append((it, melhor_val))
            sem_melhora = 0
            if verbose:
                print(
                    f"    [GVNS] it {it:4d}  violacao={melhor_val[0]:.2f}  "
                    f"distancia={melhor_val[1]:.2f}"
                )
        else:
            sem_melhora += 1

        if sem_melhora >= reinicios_sem_melhora:
            semente = rng.randrange(10**6)
            novo = rota_insercao_mais_barata(instancia, semente=semente)
            atual = list(novo) + [novo[0]]
            atual, val_atual = vnd(atual, instancia, fixo_fim=True)
            sem_melhora = 0

    return melhor[:-1], melhor_val, historico


# ---------------------------------------------------------------------------
# Interface usada pelo main.py
# ---------------------------------------------------------------------------

def resolver_classico(instancia, iteracoes=200, k_max=4, seed=42, verbose=True):
    """Executa o baseline clássico completo e devolve um dicionário no
    MESMO formato do pipeline quântico, para comparação direta."""
    if verbose:
        print("=" * 70)
        print("ALGORITMO CLÁSSICO — GVNS (critério lexicográfico)")
        print("=" * 70)
        print(
            f"[C1] Instância: {instancia['n']} nós "
            f"(depósito + {instancia['n'] - 1} clientes)"
        )

    t0 = time_module.time()

    inicial = rota_vizinho_mais_proximo(instancia)
    val_inicial = avaliar(list(inicial) + [inicial[0]], instancia)
    if verbose:
        print(f"[C2] Rota inicial (vizinho mais próximo): {inicial}")
        print(
            f"     violacao={val_inicial[0]:.2f}  distancia={val_inicial[1]:.2f}"
        )
        print(f"[C3] GVNS: {iteracoes} iterações, k_max={k_max}, seed={seed}")

    rota, valor, historico = gvns(
        instancia, iteracoes=iteracoes, k_max=k_max, seed=seed, verbose=verbose
    )
    tempo = time_module.time() - t0

    percurso = simular_percurso(rota, instancia, fechar_no_deposito=True)
    custo = custo_rota(list(rota) + [rota[0]], instancia["C"])

    return dict(
        algoritmo="classico",
        rota=rota,
        custo=custo,
        violacao=valor[0],
        viavel=not percurso["inviavel"],
        percurso=percurso,
        rota_inicial=inicial,
        valor_inicial=val_inicial,
        historico=historico,
        tcl=tempo,
        tsq=0.0,
        tte=tempo,
    )