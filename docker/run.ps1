param(
    [string]$command = "help"
)

$composeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $composeDir

$composeFiles = @("-f", "docker-compose.yml")
$null = & nvidia-smi 2>$null
if ($LASTEXITCODE -eq 0) {
    $composeFiles = @("-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml")
}

switch ($command) {
    "up" {
        Write-Host "Iniciando Elasticsearch..."
        docker compose up -d elasticsearch
        Write-Host "Elasticsearch pronto em localhost:9200"
    }
    "down" {
        Write-Host "Parando todos os servicos..."
        docker compose down
    }
    "index" {
        Write-Host "Executando pipeline de indexacao... args: $args"
        docker compose $composeFiles run --rm system python src/main_index.py @args
    }
    "corpus" {
        Write-Host "Gerando corpus.jsonl a partir dos metadados... args: $args"
        Set-Location $composeDir\..
        & python src/build_corpus.py @args
    }
    "app" {
        Write-Host "Iniciando Streamlit em http://localhost:8501..."
        docker compose $composeFiles run --rm -p 8501:8501 system streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
    }
    "shell" {
        Write-Host "Abrindo shell interativo..."
        docker compose run --rm system bash
    }
    "build" {
        $buildArgs = @()
        $envPath = Join-Path $composeDir "..\.env"
        if (Test-Path $envPath) {
            $content = Get-Content $envPath | Where-Object { $_ -match '^PYTORCH_INDEX_URL=' }
            if ($content) {
                $url = ($content -split '=', 2)[1].Trim().Trim('"').Trim("'")
                $buildArgs += "--build-arg"
                $buildArgs += "PYTORCH_INDEX_URL=$url"
                Write-Host "Usando PYTORCH_INDEX_URL=$url"
            }
        }
        Write-Host "Construindo imagem..."
        docker compose build @buildArgs system
    }
    "queries" {
        Write-Host "Gerando 2 consultas por video (factoid + keyword) via LLM... args: $args"
        Set-Location $composeDir\..
        & python src/generate_queries.py @args
    }
    "pooling" {
        Write-Host "Executando BM25 pooling... args: $args"
        docker compose $composeFiles run --rm system python src/pooling.py @args
    }
    "judge" {
        Write-Host "Executando julgamento de relevancia via LLM... args: $args"
        Set-Location $composeDir\..
        & python src/judge_relevance.py @args
    }
    "build-collection" {
        Write-Host "Montando colecao BEIR final... args: $args"
        Set-Location $composeDir\..
        & python src/build_collection.py @args
    }
    "evaluate" {
        Write-Host "Avaliando metricas TREC... args: $args"
        Set-Location $composeDir\..
        & python src/evaluate.py @args
    }
    "stats" {
        Write-Host "Estatisticas do indice ES... args: $args"
        docker compose $composeFiles run --rm system python src/stats_index.py @args
    }
    "logs" {
        docker compose logs -f
    }
    default {
        Write-Host "Uso: $($MyInvocation.MyCommand.Name) {comando} [args]"
        Write-Host ""
        Write-Host "--- Search App (requer Docker + ES) ---"
        Write-Host "  up       Inicia Elasticsearch"
        Write-Host "  down     Para todos os servicos"
        Write-Host "  build    Reconstroi a imagem Docker"
        Write-Host "  index    Pipeline de indexacao (--window, --offset, --input, --batch, --corpus)"
        Write-Host "  app      Inicia interface Streamlit (localhost:8501)"
        Write-Host "  shell    Abre bash interativo no container"
        Write-Host "  stats    Estatisticas dos videos indexados (--index, --verbose)"
        Write-Host "  logs     Segue os logs do container"
        Write-Host ""
        Write-Host "--- Colecao de Referencia (roda local, sem Docker) ---"
        Write-Host "  corpus   Gera corpus.jsonl dos metadados (--only-with-video, --metadata)"
        Write-Host "  queries  Gera consultas via LLM (--provider, --window, --offset, --force, --api-key)"
        Write-Host "  pooling  BM25 pooling dentro do container (--top-k, --variants, --merge)"
        Write-Host "  judge    Julga relevancia via LLM (--provider, --window, --batch-size, --api-key)"
        Write-Host "  build-collection  Monta qrels + hard negatives"
        Write-Host "  evaluate Avalia metricas TREC (--k, --pool-source)"
    }
}