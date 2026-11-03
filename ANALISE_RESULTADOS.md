# Análise dos resultados — QAOA (AWS SV1, sem lookup) × GVNS clássico

Comparação entre o algoritmo quântico (path-slicing + QUBO + QAOA, ângulos calibrados
com COBYLA **na AWS**, sem `--lookup`, `--busca-local nenhuma`) e o baseline clássico
(GVNS), no benchmark TSPTW `rc_20x.y` (30 instâncias, instância completa, `seed=42`,
`K=4`, 1 camada, 2000 shots, `maxiter=40`).

Arquivos analisados: `no_lookup_aws_merged.csv` (22 instâncias) e `classico_merged.csv` (30).

## Parâmetros usados: `--lookup` e `--busca-local`

### `--lookup` (origem dos ângulos do QAOA)

O QAOA precisa de dois ângulos por camada, `gamma` e `beta`, para montar o circuito de cada fatia.
Esta flag define **de onde eles vêm**:

| Modo | O que faz | Custo |
|---|---|---|
| **sem `--lookup`** (usado nesta análise) | **Calibra** os ângulos: um laço variacional roda o circuito, mede o valor esperado e o COBYLA ajusta `gamma`/`beta` (até `--maxiter` = 40 avaliações por fatia). Os ângulos encontrados são **salvos** no arquivo JSON (`--angulos`). | Alto: com `--AWS`, cada avaliação é uma tarefa paga no SV1 (por isso pede confirmação). |
| **com `--lookup`** | **Não otimiza nada.** Apenas **lê** do JSON os ângulos já calibrados antes e monta o circuito. Se faltar o ângulo de alguma fatia, dá erro em vez de calibrar. | Baixo: só a execução final (medição) de cada fatia. |

Detalhes importantes:

- Cada entrada do JSON é identificada por um **hash dos coeficientes do QUBO** (mais o número de
  camadas). Os ângulos só valem para o problema em que foram calibrados: se mudar a instância,
  o `K`, a penalidade ou a ordem decodificada da fatia anterior, o hash muda e o `--lookup` recusa.
- Na prática: calibra-se uma vez (sem `--lookup`, pagando o COBYLA) e depois repete-se o
  experimento com `--lookup`, que reaproveita os ângulos. Os arquivos `lookup_aws_rc_*.csv` vêm
  dessa segunda etapa; os `no_lookup_aws_*.csv` são a calibração completa.
- Do ponto de vista da regra do projeto, o COBYLA é a única parte clássica "no laço" do quântico;
  o `--lookup` elimina esse laço na execução (só lê valores pré-calculados).

### `--busca-local` (quanta busca local clássica entra no pipeline quântico)

Após o QAOA resolver as fatias, é possível refinar a rota com busca local **clássica** (VND).
Esta flag controla o quanto:

| Valor | O que faz |
|---|---|
| **`nenhuma`** (padrão; usado nesta análise) | Só o resultado do QAOA em cada fatia, sem refinamento. É a **comparação limpa** contra o clássico: tudo o que a rota tem de bom vem do pipeline quântico. |
| `fatia` | Aplica VND **dentro de cada fatia**, depois do QAOA. |
| `completa` | VND por fatia **mais** um VND global sobre a rota inteira. Faz o pipeline convergir para o mesmo ótimo local do baseline clássico, então a comparação deixa de medir o quântico. |

Por isso, `nenhuma` é o valor coerente com a regra do projeto (parte quântica sem componente
clássico além do pré-processamento); `fatia` e `completa` acrescentam trabalho clássico real e
exigiriam aprovação explícita. Também é o motivo de o gap de qualidade nesta análise ser alto:
sem busca local, o resultado fica limitado pela rota inicial e pela divisão em fatias.

## 1. Cobertura

| Item | Resultado |
|---|---|
| Instâncias no benchmark | 30 |
| Quântico concluído (AWS, sem lookup) | **22** |
| Quântico **falhou** | **8**: `202.1`, `202.3`, `203.2`, `204.1`, `205.3`, `206.2`, `207.2`, `208.1` |
| Clássico concluído | 30 (todas viáveis) |

As 8 falhas têm a mesma causa e são determinísticas: uma sub-rota (fatia) é **inviável** —
"nenhuma ordem dos clientes cumpre as janelas" — e o erro aponta para a rota inicial /
composição das fatias, não para o QAOA. Como a rota inicial é a mesma, repetir a execução
falharia igual (e geraria nova cobrança na AWS). Nessas 8 instâncias o clássico acha solução
viável. Ou seja, **o pipeline quântico cobre 73% das instâncias; o clássico, 100%.**

## 2. Qualidade das soluções

Todas as 22 soluções quânticas são **viáveis** (violação de janelas = 0), assim como as do clássico.
A métrica abaixo é a distância total; gap = (quântico / clássico − 1) × 100.

| Estatística (22 instâncias) | Valor |
|---|---|
| Gap médio | **+32,5 %** |
| Gap mediano | +28,5 % |
| Menor gap | 0,0 % (`rc_207.4`, 6 nós) |
| Maior gap | +79,8 % (`rc_208.2`) |
| Instâncias em que o quântico empata/vence | 1 (`rc_207.4`) |
| Instâncias com gap > 50 % | 5: `202.2` (+65 %), `203.4` (+76 %), `204.2` (+65 %), `207.3` (+60 %), `208.2` (+80 %) |

Destaques:

- O clássico **venceu em 21 das 22** instâncias; o quântico só empatou na menor delas (6 nós).
  Em `rc_206.1` (4 nós) a diferença é de 0,9 % (88,62 × 87,85).
- As melhores aproximações são instâncias pequenas ou de gap baixo: `201.4` (+0,4 %),
  `206.1` (+0,9 %), `206.4` (+11,2 %), `201.3` (+10,0 %), `205.4` (+11,1 %).
- A razão de aproximação média por fatia reportada é 1,0 em todas as instâncias: **cada fatia
  é resolvida de forma ótima localmente** (todas as decodificações válidas), e portanto a
  perda de qualidade vem da **decomposição em fatias** (path-slicing sobre a rota inicial,
  sem busca local global), e não de o QAOA errar a solução da fatia.
- Sem busca local (`nenhuma`), o quântico fica preso à vizinhança da rota inicial; o GVNS
  explora muito mais o espaço de busca. A comparação é "limpa", mas favorece o clássico.

## 3. Tempo

Tempos em segundos, do CSV (`tte_s` = tempo total; `tsq_s` = tempo de simulação quântica;
`tcl_s` = tempo clássico).

| Medida | Quântico (22 inst.) | Clássico (mesmas 22) | Clássico (30) |
|---|---|---|---|
| Tempo total somado | 22 224 s ≈ **6,2 h** | 13 487 s ≈ 3,7 h | 23 822 s ≈ 6,6 h |
| Razão quântico/clássico (mediana) | **≈ 2,5×** mais lento | — | — |

- O quântico foi **mais rápido em apenas 4 instâncias**, todas grandes: `206.4` (0,47×),
  `208.3` (0,88×), `207.1` (0,93×), `203.3` (0,95×). Nas pequenas/médias o clássico é muito
  mais rápido (ex.: `202.2`: 33 s × 685 s; `207.4`: 0,9 s × 205 s).
- O tempo do clássico **cresce rápido com o tamanho** (até 3 064 s em `206.4`, 38 nós), enquanto
  o quântico cresce de forma mais suave (555 s a 2 117 s), pois depende sobretudo do número
  de fatias × avaliações do COBYLA.
- Quase todo o tempo quântico é **latência de rede/fila da AWS**, não computação: a
  "execução no simulador" (tempo de execução do SV1) soma só ≈ 94 s no total, contra 22 224 s de
  `tte_s` — cerca de 290× menos. O tempo classico do pipeline quântico (pré-processamento) é
  desprezível (≤ 0,015 s por instância).
- Portanto **o tempo quântico medido não representa o tempo de um computador quântico real**:
  reflete o modelo de execução (uma tarefa SV1 por avaliação do COBYLA, em série).
  Os tempos do clássico dependem da máquina local e da implementação em Python.

## 4. Recursos

- **Tarefas na AWS:** 183 fatias no total (7 a 13 por instância) × (40 avaliações COBYLA + 1 medição
  final) ≈ **7 500 tarefas SV1** (limite superior; o log mostra, por exemplo, 298 tarefas em
  `rc_201.2`, que tem 9 fatias).
- **Custo (SV1, US$ 0,075/min, mínimo de 3 s por tarefa):** estimativa superior de
  ≈ **US$ 28** para as 22 instâncias concluídas (≈ 375 min faturados). Isso é uma estimativa
  pelo mínimo de cobrança, não a fatura real; confira o valor no console da AWS/Billing.
  Execuções repetidas por causa dos encerramentos por falta de memória (ver abaixo) e as
  8 instâncias que falharam podem ter gerado custo adicional parcial.
- **Ângulos:** 237 entradas salvas em `angulos_no_lookup_aws.json` (arquivo separado, para
  não alterar o `angulos.json` usado pelo modo `--lookup`). O custo de calibrar é **pago uma vez**:
  com `--lookup`, as mesmas instâncias rodam sem o laço COBYLA.
- **Tamanho do problema:** cada fatia usa K = 4 (QUBO de poucos qubits, `K−2` clientes
  internos), então o simulador SV1 nunca é o gargalo de memória; o limite é o número de
  tarefas/latência.
- **Máquina local:** a execução em lote foi interrompida duas vezes pelo Claude Code por
  pouca memória RAM (≈ 3 GB livres de 16 GB na checagem); cada processo Python usa ≈ 200 MB.
  Os laços sobreviventes geraram execuções duplicadas, tratadas na junção (deduplicação por instância).

## 5. Pontos de atenção

1. **Regra do projeto (CLAUDE.md):** o modo sem lookup usa **COBYLA, um otimizador clássico**, para
   calibrar os ângulos — um laço híbrido quântico-clássico. A regra do projeto só permite isso com
   aprovação explícita; já existia no código (módulo `qaoa_calibracao.py`, isolado), mas vale
   confirmar que o uso está aprovado para o TCC. O modo `--lookup` apenas lê ângulos.
2. **Falhas de viabilidade (8/30):** melhorar a rota inicial / composição das fatias (ou
   reparar a fatia inviável) é o ajuste com maior retorno para a cobertura.
3. **Qualidade:** habilitar `--busca-local fatia` ou `completa` aproxima o resultado do
   clássico, mas introduz componente clássico no pipeline (fora da regra do projeto sem aprovação).
4. A coluna `gap_pct_vs_classico` dos CSVs está vazia (rodadas separadas); o gap acima foi
   calculado a partir de `distancia_total`.
5. Mesma `seed` (42) e uma única execução por instância: não há variância medida, então diferenças
   pequenas (ex.: `201.4`, `206.1`) não são estatisticamente conclusivas.

## 6. Conclusão

Com o pipeline atual (sem busca local, K = 4, 1 camada) o QAOA resolve **cada fatia de forma
ótima**, mas a solução global fica em média **≈ 32 % pior** que o GVNS, cobre **22 de 30**
instâncias e é **≈ 2,5× mais lento** (dominado por latência da AWS), a um custo estimado
de **≈ US$ 28** em SV1. O quântico só compensa em tempo nas maiores instâncias testadas.
