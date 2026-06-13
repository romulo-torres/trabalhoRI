param(
    [string]$command = "help"
)

$composeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $composeDir

$composeFiles = @("-f", "docker-compose.yml")

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
        Write-Host "Executando gerador de queries (LLM)..."
        Set-Location $composeDir\..
        & python src/generate_queries.py @args
    }
    "stats" {
        Write-Host "Estatisticas do indice ES... args: $args"
        docker compose $composeFiles run --rm system python src/stats_index.py @args
    }
    "logs" {
        docker compose logs -f
    }
    default {
        Write-Host "Uso: $($MyInvocation.MyCommand.Name) {up|down|index|app|shell|build|queries|stats|logs}"
        Write-Host ""
        Write-Host "  up       - Inicia Elasticsearch"
        Write-Host "  down     - Para todos os servicos"
        Write-Host "  index    - Executa pipeline de indexacao (aceita: --window, --offset, --input, --batch)"
        Write-Host "  app      - Inicia interface Streamlit"
        Write-Host "  shell    - Abre shell interativo no container"
        Write-Host "  build    - Reconstroi a imagem Docker"
        Write-Host "  queries  - Gera queries de busca via LLM (local, fora do Docker)"
        Write-Host "  stats    - Estatisticas dos videos indexados (aceita: --index, --verbose)"
        Write-Host "  logs     - Segue os logs"
    }
}
