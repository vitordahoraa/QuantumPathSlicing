"""
braket_dispositivo.py
===========================================================================
Escolha do dispositivo onde os circuitos rodam: o LocalSimulator (na sua
máquina, gratuito) ou um simulador gerenciado do Amazon Braket (AWS, cobrado).

Nada aqui é algoritmo: é só infraestrutura de execução. A configuração da
AWS (credenciais, região, bucket S3) vem de variáveis de ambiente, que podem
ser carregadas de um arquivo ".env" na pasta do projeto (ver .env.example).
O ".env" está no .gitignore e não deve ser versionado.
"""

import os
from pathlib import Path

import numpy as np

ARN_SV1 = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
PREFIXO_S3_PADRAO = "qaoa-tsptw"

RAIZ_PROJETO = Path(__file__).resolve().parent


class ConfiguracaoAwsError(RuntimeError):
    """Configuração da AWS ausente ou incompleta."""


def carregar_env(caminho=None):
    """Carrega pares CHAVE=VALOR de um arquivo .env para os.environ.

    Variáveis que já existem no ambiente têm prioridade sobre o arquivo.
    Aceita comentários (#), linhas em branco e valores entre aspas. Se o
    arquivo não existir, não faz nada. Devolve True se carregou algum."""
    caminho = Path(caminho) if caminho else RAIZ_PROJETO / ".env"
    if not caminho.is_file():
        return False

    carregou = False
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        chave, valor = chave.strip(), valor.strip()
        if len(valor) >= 2 and valor[0] == valor[-1] and valor[0] in "\"'":
            valor = valor[1:-1]
        if chave and valor and chave not in os.environ:
            os.environ[chave] = valor
            carregou = True
    return carregou


class DispositivoBraket:
    """Interface única para os dois tipos de dispositivo: devolve o vetor de
    probabilidades das 2^n bitstrings de um circuito.

    Também soma o TEMPO DE EXECUÇÃO NO SIMULADOR que o próprio serviço
    reporta nos metadados do resultado (simulatorMetadata.executionDuration,
    em milissegundos). É só o tempo de simulação: não inclui rede, fila,
    nem gravação/download do resultado, que entram no tempo de relógio
    medido por quem chama. O LocalSimulator não reporta esse campo."""

    def __init__(self, device, nome, s3=None):
        self.device = device
        self.nome = nome
        self.s3 = s3  # (bucket, prefixo) — só para dispositivos AWS
        self.tarefas = 0
        self.tarefas_com_execucao = 0
        self.execucao_s = 0.0

    def probabilidades(self, circ, shots):
        kwargs = {"s3_destination_folder": self.s3} if self.s3 else {}
        tarefa = self.device.run(circ, shots=shots, **kwargs)
        resultado = tarefa.result()

        self.tarefas += 1
        sim = getattr(resultado.additional_metadata, "simulatorMetadata", None)
        ms = getattr(sim, "executionDuration", None)
        if ms is not None:
            self.tarefas_com_execucao += 1
            self.execucao_s += float(ms) / 1000.0

        return np.asarray(resultado.values[0])

    def instantaneo(self):
        """Contadores atuais, para medir um trecho com desde()."""
        return (self.tarefas, self.tarefas_com_execucao, self.execucao_s)

    def desde(self, base):
        """Tarefas e tempo de execução acumulados desde o instantâneo 'base'.
        'execucao_s' é None se nenhuma tarefa do trecho reportou o tempo."""
        tarefas, com_exec, exec_s = base
        reportou = self.tarefas_com_execucao - com_exec > 0
        return dict(
            tarefas=self.tarefas - tarefas,
            execucao_s=(self.execucao_s - exec_s) if reportou else None,
        )


def criar_dispositivo(aws=False):
    """LocalSimulator (aws=False) ou simulador gerenciado do Braket (aws=True).

    Para a AWS, lê do ambiente (e do .env):
      BRAKET_S3_BUCKET   obrigatória
      BRAKET_S3_PREFIX   opcional (padrão: qaoa-tsptw)
      BRAKET_DEVICE_ARN  opcional (padrão: SV1)
      AWS_REGION         região (ou AWS_DEFAULT_REGION)
      credenciais        AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, ou
                         AWS_PROFILE (lidas pelo boto3)"""
    try:
        if aws:
            from braket.aws import AwsDevice, AwsSession
            import boto3
        else:
            from braket.devices import LocalSimulator
    except ImportError as exc:
        raise ImportError(
            "O Amazon Braket SDK é necessário para o algoritmo quântico.\n"
            "Instale com:  pip install amazon-braket-sdk\n"
            f"Erro original: {exc}"
        ) from exc

    if not aws:
        return DispositivoBraket(LocalSimulator(), "LocalSimulator")

    carregar_env()
    bucket = os.environ.get("BRAKET_S3_BUCKET")
    if not bucket:
        raise ConfiguracaoAwsError(
            "BRAKET_S3_BUCKET não definido. Crie o arquivo .env a partir de "
            ".env.example e informe o bucket S3 dos resultados do Braket."
        )
    prefixo = os.environ.get("BRAKET_S3_PREFIX", PREFIXO_S3_PADRAO)
    arn = os.environ.get("BRAKET_DEVICE_ARN", ARN_SV1)
    regiao = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

    sessao = AwsSession(boto_session=boto3.Session(region_name=regiao))
    device = AwsDevice(arn, aws_session=sessao)
    return DispositivoBraket(device, f"AWS {arn.rsplit('/', 1)[-1]}",
                             s3=(bucket, prefixo))
