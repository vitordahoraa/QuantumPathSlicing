"""
main.py
===========================================================================
Ponto de entrada da avaliação de desempenho de uma abordagem híbrida
quântica de busca local aplicada ao TSPTW.

Responsabilidades deste arquivo:
  1. carregamento/parsing/validação da instância de benchmark;
  2. interface de linha de comando (qual algoritmo rodar e com quais
     parâmetros);
  3. execução e relatório comparativo dos resultados.

Os algoritmos em si ficam em:
  - classic_algorithm.py      (GVNS + função objetivo lexicográfica)
  - quantum_path_slicing.py   (Path-Slicing + QUBO + QAOA no Braket)

Uso:
    python main.py --algorithm ambos
    python main.py --algorithm classico  --instancia benchmark/rc_201.2.txt
    python main.py --algorithm quantico  --clientes 6 --K 4 --camadas 1

Requisitos:
    pip install amazon-braket-sdk numpy scipy
    (o SDK do Braket só é necessário para --algorithm quantico/ambos)

---------------------------------------------------------------------------
FORMATO DO ARQUIVO DE BENCHMARK
---------------------------------------------------------------------------
Formato TSPTW baseado em MATRIZ DE TEMPOS DE VIAGEM (Dumas / Gendreau /
derivados de Solomon, ex.: "rc_201.2.txt"), em vez de coordenadas:

    linha 1   : n (número de nós, incluindo o depósito 0)
    n linhas  : matriz n x n de tempos de viagem M[i][j]
    n linhas  : janela de tempo de cada nó, "a_i  b_i"

Convenção destes arquivos: o tempo de serviço do nó i já está EMBUTIDO na
linha i da matriz, isto é

    M[i][j] = d(i, j) + s_i        e        M[i][i] = s_i

(é por isso que a diagonal vale 10 para os clientes e 0 para o depósito, e
que M[0][1] = 33.541 enquanto M[1][0] = 43.541 — a diferença é exatamente
s_1 = 10). Como o restante do código soma o serviço explicitamente
(chegada = t + servico_i + C[i][j]), o parser SEPARA as duas parcelas:
extrai s_i da diagonal e devolve C[i][j] = M[i][j] - s_i, que é a matriz de
tempos de viagem puros — simétrica, como esperado.
"""

import argparse
import os
import sys

import numpy as np

import classic_algorithm as classico
import quantum_path_slicing as quantico


# ---------------------------------------------------------------------------
# Localização dos arquivos de benchmark
# ---------------------------------------------------------------------------

# Diretório (relativo a ESTE arquivo .py) onde ficam os .txt de benchmark, e
# caminho relativo da instância usada por padrão. Basta trocar a string
# abaixo — ou passar --instancia na linha de comando — para rodar outra.
PASTA_BENCHMARK = "benchmark"
INSTANCIA_PADRAO = "benchmark/rc_206.3.txt"

# raiz usada para resolver caminhos relativos: a pasta onde este script está
# salvo (e não o diretório de trabalho atual), para que funcione mesmo
# sendo chamado de outro lugar, ex.: `python ~/tcc/main.py`
RAIZ_SCRIPT = os.path.dirname(os.path.abspath(__file__))


def resolver_caminho_instancia(caminho):
    """Converte um caminho relativo (ex.: "benchmark/rc_201.2.txt" ou
    apenas "rc_201.2.txt") no caminho absoluto do arquivo, procurando:

        1. o caminho exatamente como foi dado (absoluto ou relativo ao cwd);
        2. o caminho relativo à pasta deste script;
        3. o caminho dentro de <pasta do script>/PASTA_BENCHMARK.

    Levanta FileNotFoundError listando os locais tentados."""
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
        with open(resolver_caminho_instancia(texto.strip()), "r", encoding="utf-8") as arq:
            return arq.read()
    return texto


# ---------------------------------------------------------------------------
# Parsing e validação
# ---------------------------------------------------------------------------

def parse_instancia_tsptw(fonte):
    """Ingestão e parsing do arquivo no formato de matriz, com validação
    estrutural (contagem de tokens, janelas a_i <= b_i, tempos não
    negativos).

    Retorna (M, C, A, B, servico, n), com M a matriz original do arquivo,
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
    """Monta o dicionário consumido pelos dois algoritmos, reindexando os
    nós em 0..n-1. O campo 'id_original' preserva o índice do nó no arquivo
    de benchmark, útil para reportar a solução na numeração original."""
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
    return dict(
        nos=nos,
        C=np.asarray(C, dtype=float),
        A=np.asarray(A, dtype=float),
        B=np.asarray(B, dtype=float),
        servico=np.asarray(servico, dtype=float),
        n=n,
    )


def carregar_instancia(fonte=INSTANCIA_PADRAO, n_clientes=None):
    """Carrega, valida e (opcionalmente) recorta a instância.

    'fonte' pode ser o caminho do arquivo — absoluto ou relativo, ex.:
    "benchmark/rc_201.2.txt" — ou o próprio conteúdo do arquivo.

    Se n_clientes for informado, o recorte é direto: depósito + os
    n_clientes PRIMEIROS clientes (ordem de leitura / ID), extraindo a
    submatriz correspondente. Não há pré-filtro de viabilidade — o
    subconjunto pode não admitir nenhuma ordem viável, e isso é detectado
    e reportado na simulação do percurso, ao final."""
    _, C, A, B, servico, n = parse_instancia_tsptw(fonte)

    indices = list(range(n))
    if n_clientes is not None:
        disponiveis = n - 1
        if n_clientes > disponiveis:
            print(
                f"[aviso] Foram solicitados {n_clientes} clientes, mas a "
                f"instância só possui {disponiveis}; usando todos."
            )
            n_clientes = disponiveis
        indices = [0] + list(range(1, 1 + n_clientes))

    idx = np.array(indices, dtype=int)
    return _montar_instancia(C[np.ix_(idx, idx)], A[idx], B[idx], servico[idx], idx)


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

TITULOS = {
    "classico": "CLÁSSICO (GVNS)",
    "quantico": "QUÂNTICO (Path-Slicing + QAOA)",
}


def _titulo(resultado):
    base = TITULOS[resultado["algoritmo"]]
    if resultado["algoritmo"] == "quantico":
        return f"{base} (busca local: {resultado['busca_local']})"
    return base


def relatar(resultado, instancia):
    """Imprime o relatório detalhado de um resultado."""
    percurso = resultado["percurso"]
    titulo = _titulo(resultado)

    print("\n" + "=" * 70)
    print(f"RESULTADO — {titulo}")
    print("=" * 70)
    print(f"Rota planejada:            {resultado['rota']}")
    print(f"Distância total:           {resultado['custo']:.2f}")
    print(f"Violação de janelas:       {resultado['violacao']:.2f}")
    print(f"Situação:                  {'VIÁVEL' if resultado['viavel'] else 'INVIÁVEL'}")

    print("\n" + "-" * 70)
    print("EXECUÇÃO DO PERCURSO (tentativa nó a nó, com o relógio real)")
    print("-" * 70)
    caminho = list(percurso["percorrido"])
    if not percurso["inviavel"]:
        caminho = caminho + [resultado["rota"][0]]
    print(f"Caminho percorrido:        {caminho}")
    print(
        "Tempos de chegada:         "
        + ", ".join(
            f"{no}@{t:.1f}" for no, t in zip(percurso["percorrido"], percurso["tempos"])
        )
    )
    print(f"Último nó visitado:        {percurso['percorrido'][-1]}")
    print(
        f"Distância percorrida:      "
        f"{classico.custo_rota(caminho, instancia['C']):.2f}"
    )

    if percurso["inviavel"]:
        no = percurso["no_falha"]
        print(
            f"Parou em:                  nó {no} "
            f"(chegada {percurso['chegada_falha']:.1f} > b_{no} = "
            f"{instancia['B'][no]:.1f})"
        )
        print(f"Nós NÃO visitados:         {percurso['nao_visitados']}")
        print(f"Qtd. de nós não visitados: {len(percurso['nao_visitados'])}")
    else:
        print("Nós NÃO visitados:         [] (todos os nós foram atendidos)")
        print(f"Retorno ao depósito em:    {percurso['retorno_deposito']:.1f}")

    if resultado.get("resumo_quantico"):
        _relatar_diagnostico_quantico(resultado)

    print("\n" + "-" * 70)
    print(f"Tempo Clássico (TCL):           {resultado['tcl']:.3f} s")
    print(f"Tempo Simulação Quântica (TSQ): {resultado['tsq']:.3f} s")
    print(f"Tempo Total de Execução (TTE):  {resultado['tte']:.3f} s")


def _relatar_diagnostico_quantico(resultado):
    """Métricas que mostram o quanto a etapa quântica de fato contribuiu.

    Sem isso é impossível distinguir 'o QAOA acertou a rota' de 'o VND
    clássico consertou tudo depois' — os dois produzem o mesmo número
    final."""
    resumo = resultado["resumo_quantico"]

    print("\n" + "-" * 70)
    print("DIAGNÓSTICO DA ETAPA QUÂNTICA")
    print("-" * 70)
    print(
        f"Decodificações fiéis:      "
        f"{resumo['decodificacoes_validas']}/{resumo['fatias']} fatias "
        f"({resumo['taxa_validas'] * 100:.0f}%) — permutação válida medida"
    )
    print(
        f"Posições vindas do QAOA:   "
        f"{resumo['posicoes_decididas_pelo_qaoa']}/{resumo['posicoes_totais']} "
        f"({resumo['taxa_posicoes_qaoa'] * 100:.0f}%) — o resto foi desempate "
        f"ou reparo guloso"
    )
    if resumo["razao_aproximacao_media"] is not None:
        print(
            f"Razão de aproximação:      "
            f"{resumo['razao_aproximacao_media']:.3f} "
            f"(1.000 = QAOA achou o ótimo exato do QUBO)"
        )
    if resumo["otimos_qubo_invalidos"]:
        print(
            f"[!] Em {resumo['otimos_qubo_invalidos']} fatia(s) o ótimo do "
            f"QUBO não é permutação válida: penalidade subdimensionada, "
            f"aumente --penalidade."
        )

    cru, pre = resultado["valor_cru"], resultado["valor_pre_refino"]
    pos = (resultado["violacao"], resultado["custo"])
    print("\nOnde cada rota foi decidida:")
    print(f"  1. só QAOA (sem busca local) {str(resultado['rota_crua']):<32}"
          f"viol={cru[0]:.2f} dist={cru[1]:.2f}")
    print(f"  2. + VND por fatia           {str(resultado['rota_pre_refino']):<32}"
          f"viol={pre[0]:.2f} dist={pre[1]:.2f}")
    print(f"  3. + VND global (final)      {str(list(resultado['rota'])):<32}"
          f"viol={pos[0]:.2f} dist={pos[1]:.2f}")
    print(f"   (busca local clássica no pipeline: {resultado['busca_local']})")
    if resultado["rota_crua"] == list(resultado["rota"]):
        print("   -> a rota final é exatamente a do QAOA: a busca local não mexeu")
    elif resultado["rota_pre_refino"] == list(resultado["rota"]):
        print("   -> o VND por fatia alterou a rota do QAOA; o refino global não mexeu")
    else:
        print("   -> a busca local clássica alterou a rota do QAOA em ambas as etapas")

    print("\nPor fatia:")
    print(f"  {'fatia':<22}{'alpha':>9}{'decodif.':>12}{'razao aprox':>14}{'rank':>14}")
    for d in resultado["diagnostico"]:
        info, met = d["decodificacao"], d["metricas"]
        dec = "fiel" if info["valida"] else f"reparo {info['reparos']}/{info['posicoes']}"
        razao = f"{met['razao_aproximacao']:.3f}" if met else "-"
        rank = f"{met['rank']}/{met['total_estados']}" if met else "-"
        print(f"  {str(d['fatia']):<22}{d['alpha']:>9.1f}{dec:>12}{razao:>14}{rank:>14}")


def comparar(res_classico, res_quantico):
    """Tabela comparativa entre os dois algoritmos, com o Gap % do
    híbrido em relação ao baseline clássico."""
    print("\n" + "=" * 70)
    print("COMPARAÇÃO")
    print("=" * 70)
    print(f"{'Métrica':<28}{'Clássico':>20}{'Quântico':>20}")
    print("-" * 70)
    print(
        f"{'Distância total':<28}{res_classico['custo']:>20.2f}"
        f"{res_quantico['custo']:>20.2f}"
    )
    print(
        f"{'Violação de janelas':<28}{res_classico['violacao']:>20.2f}"
        f"{res_quantico['violacao']:>20.2f}"
    )
    print(
        f"{'Viável':<28}"
        f"{('sim' if res_classico['viavel'] else 'não'):>20}"
        f"{('sim' if res_quantico['viavel'] else 'não'):>20}"
    )
    print(
        f"{'Nós atendidos':<28}"
        f"{len(res_classico['percurso']['percorrido']):>20}"
        f"{len(res_quantico['percurso']['percorrido']):>20}"
    )
    print(f"{'TCL (s)':<28}{res_classico['tcl']:>20.3f}{res_quantico['tcl']:>20.3f}")
    print(f"{'TSQ (s)':<28}{res_classico['tsq']:>20.3f}{res_quantico['tsq']:>20.3f}")
    print(f"{'TTE (s)':<28}{res_classico['tte']:>20.3f}{res_quantico['tte']:>20.3f}")
    print("-" * 70)

    # o Gap só é comparável entre soluções de mesma viabilidade
    base = res_classico["custo"]
    if base > 0:
        gap = (res_quantico["custo"] - base) / base * 100
        print(f"Gap (%) do híbrido vs. baseline clássico: {gap:+.2f}%")
    if res_classico["viavel"] != res_quantico["viavel"]:
        print(
            "[!] Atenção: as soluções têm viabilidade diferente — o Gap de "
            "distância não é diretamente comparável."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def montar_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Avaliação de uma abordagem híbrida quântica de busca local "
            "aplicada ao TSPTW, comparada a um baseline clássico (GVNS)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemplos:\n"
            "  python main.py --algorithm ambos\n"
            "  python main.py --algorithm classico --instancia benchmark/rc_201.2.txt\n"
            "  python main.py --algorithm quantico --clientes 6 --K 4 --camadas 2\n"
        ),
    )
    p.add_argument(
        "--algorithm", "-a",
        choices=["classico", "quantico", "ambos"],
        default="ambos",
        help="qual algoritmo executar (padrão: ambos)",
    )
    p.add_argument(
        "--instancia", "-i",
        default=INSTANCIA_PADRAO,
        help=f"caminho do arquivo de benchmark (padrão: {INSTANCIA_PADRAO})",
    )
    p.add_argument(
        "--clientes", "-c", type=int, default=4,
        help="recorte: nº de clientes após o depósito (padrão: 4; use 0 para todos)",
    )
    p.add_argument("--seed", type=int, default=42, help="semente aleatória (padrão: 42)")
    p.add_argument("--quiet", "-q", action="store_true", help="suprime o log de progresso")

    g = p.add_argument_group("parâmetros do algoritmo quântico")
    g.add_argument("--K", type=int, default=4, help="tamanho da fatia no path-slicing (padrão: 4)")
    g.add_argument("--camadas", type=int, default=1, help="camadas p do QAOA (padrão: 1)")
    g.add_argument("--amostras", type=int, default=2000, help="shots por execução (padrão: 2000)")
    g.add_argument("--maxiter", type=int, default=40, help="iterações do COBYLA (padrão: 40)")
    g.add_argument(
        "--penalidade", type=float, default=1.3,
        help=(
            "fator de escala de alpha/beta sobre a maior aresta da fatia "
            "(padrão: 1.3; é auto-calibrado para cima se o mínimo do QUBO "
            "deixar de ser uma permutação válida)"
        ),
    )
    g.add_argument(
        "--busca-local", dest="busca_local",
        choices=["nenhuma", "fatia", "completa"], default="nenhuma",
        help=(
            "quanta busca local CLÁSSICA entra no pipeline quântico. "
            "nenhuma (padrão) = comparação limpa contra o clássico; "
            "fatia = VND dentro de cada fatia; completa = VND por fatia + "
            "VND global, que faz o pipeline convergir para o mesmo ótimo "
            "local do baseline"
        ),
    )
    g.add_argument(
        "--construcao", choices=classico.CONSTRUCOES, default="auto",
        help=(
            "heurística da rota inicial (pré-processamento): auto = a melhor "
            "entre vizinho, prazo, inicio e insercao (padrão)"
        ),
    )
    g.add_argument(
        "--sem-metricas", dest="metricas", action="store_false",
        help="pula a enumeração exaustiva do QUBO (diagnóstico mais rápido)",
    )
    g.add_argument(
        "--exato", action="store_true",
        help=(
            "avalia o valor esperado com a distribuição exata (shots=0) "
            "durante a otimização dos ângulos; a medição final continua "
            "amostrada com --amostras"
        ),
    )

    g = p.add_argument_group("parâmetros do algoritmo clássico")
    g.add_argument("--iteracoes", type=int, default=200, help="iterações do GVNS (padrão: 200)")
    g.add_argument("--kmax", type=int, default=4, help="intensidade máxima do shaking (padrão: 4)")

    return p


def main(argv=None):
    args = montar_parser().parse_args(argv)
    verbose = not args.quiet

    # 1) Carregamento da instância
    try:
        caminho = (
            resolver_caminho_instancia(args.instancia)
            if _parece_caminho(args.instancia)
            else "<conteúdo inline>"
        )
        n_clientes = args.clientes if args.clientes > 0 else None
        instancia = carregar_instancia(args.instancia, n_clientes=n_clientes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("TSPTW — ABORDAGEM HÍBRIDA QUÂNTICA DE BUSCA LOCAL")
    print("=" * 70)
    print(f"Arquivo:   {caminho}")
    print(f"Instância: {instancia['n']} nós (depósito + {instancia['n'] - 1} clientes)")
    print(f"Algoritmo: {args.algorithm}")
    print()

    res_classico = res_quantico = None

    # 2) Execução
    if args.algorithm in ("classico", "ambos"):
        res_classico = classico.resolver_classico(
            instancia,
            iteracoes=args.iteracoes,
            k_max=args.kmax,
            seed=args.seed,
            verbose=verbose,
        )
        relatar(res_classico, instancia)

    if args.algorithm in ("quantico", "ambos"):
        if not quantico.BRAKET_DISPONIVEL:
            print(
                "\n[erro] O Amazon Braket SDK não está instalado; não é "
                "possível rodar o algoritmo quântico.\n"
                "        pip install amazon-braket-sdk",
                file=sys.stderr,
            )
            return 1
        print()
        try:
            res_quantico = quantico.resolver_quantico(
                instancia,
                K=args.K,
                camadas_qaoa=args.camadas,
                amostras=args.amostras,
                maxiter=args.maxiter,
                seed=args.seed,
                fator_penalidade=args.penalidade,
                metricas=args.metricas,
                exato=args.exato,
                busca_local=args.busca_local,
                construcao=args.construcao,
                verbose=verbose,
            )
        except quantico.FatiaInviavelError as exc:
            print(f"\n[erro] {exc}", file=sys.stderr)
            return 2
        relatar(res_quantico, instancia)

    # 3) Comparação
    if args.algorithm == "ambos":
        comparar(res_classico, res_quantico)

    return 0


if __name__ == "__main__":
    sys.exit(main())