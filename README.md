# Uso dos Scripts `run.sh` e `run.ps1`

## Visão Geral

Os scripts `docker/run.sh` (Linux) e `docker/run.ps1` (Windows/PowerShell) controlam todas as operações do sistema via Docker. Exceto `queries`, todos os comandos rodam dentro do container.

## Passo a passo Linux

1. Construir a imagem Docker

```bash
./docker/run.sh build
```

2. Subir o Elasticsearch

```bash
./docker/run.sh up
```

3. Iniciar interface web

```bash
./docker/run.sh app
```

4. Indexar vídeos (ex: 10 primeiros)

```bash
./docker/run.sh index -w 10
```

5. Gerar consultas de busca via LLM (roda no host)

```bash
./docker/run.sh queries -w 10 --provider lm-studio -k http://localhost:1234/v1
```

6. Ver estatísticas do índice

```bash
./docker/run.sh stats
```

7. (opcional) Acompanhar logs

```bash
./docker/run.sh logs
```

8. Parar tudo

```bash
./docker/run.sh down
```

## Passo a passo Windows

1. Construir a imagem Docker

```powershell
.\docker\run.ps1 build
```

2. Subir o Elasticsearch

```powershell
.\docker\run.ps1 up
```

3. Iniciar interface web

```powershell
.\docker\run.ps1 app
```

4. Indexar vídeos (ex: 10 primeiros)

```powershell
.\docker\run.ps1 index -w 10
```

5. Gerar consultas de busca via LLM (roda no host)

```powershell
.\docker\run.ps1 queries -w 10 --provider lm-studio -k http://localhost:1234/v1
```

6. Ver estatísticas do índice

```powershell
.\docker\run.ps1 stats
```

7. (opcional) Acompanhar logs

```powershell
.\docker\run.ps1 logs
```

8. Parar tudo
```powershell
.\docker\run.ps1 down
```

## Comandos com parâmetros

### `index`

Executa o pipeline de indexação. Aceita os mesmos argumentos de `main_index.py`.

```bash
./docker/run.sh index                                    # Todos os pendentes
```

```bash
./docker/run.sh index -w 10                               # 10 novos vídeos
```

```bash
./docker/run.sh index -o 100 -w 50                        # 50 a partir do 100
```

```bash
./docker/run.sh index -i data/metadata/videos_metadata.json -w 10  # Com input alternativo
```

```bash
./docker/run.sh index --batch                             # Modo batch (mais RAM)
```
  
```powershell
.\docker\run.ps1 index                                    # Todos os pendentes
```

```powershell
.\docker\run.ps1 index -w 10                               # 10 novos vídeos
```

```powershell
.\docker\run.ps1 index -o 100 -w 50                        # 50 a partir do 100
```

```powershell
.\docker\run.ps1 index -i data/metadata/videos_metadata.json -w 10
```

```powershell
.\docker\run.ps1 index --batch
```

### `build`

Reconstrói a imagem Docker. Se o `.env` contiver `PYTORCH_INDEX_URL`, a imagem é construída com suporte CUDA.

```bash
./docker/run.sh build
```

```powershell
.\docker\run.ps1 build
```

### `queries`

Gera consultas de busca via LLM. Roda **fora do Docker** porque não precisa de torch nem ES. Aceita os mesmos argumentos de `generate_queries.py`.

```bash
./docker/run.sh queries -w 10 --provider openai
```

```bash
./docker/run.sh queries -w 10 --provider lm-studio -k http://localhost:1234/v1
```

```bash
./docker/run.sh queries --force
```

```powershell
.\docker\run.ps1 queries -w 10 --provider openai
```

```powershell
.\docker\run.ps1 queries -w 10 --provider lm-studio -k http://localhost:1234/v1
```

```powershell
.\docker\run.ps1 queries --force
```