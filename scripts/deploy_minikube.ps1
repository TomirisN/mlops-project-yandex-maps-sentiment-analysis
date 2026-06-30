# Deploy to Minikube (Windows PowerShell)
# Usage: .\scripts\deploy_minikube.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$MinikubeCandidates = @(
    "$env:USERPROFILE\.bin\minikube.exe",
    "$env:ProgramFiles\Kubernetes\Minikube\minikube.exe",
    "$env:LOCALAPPDATA\Programs\minikube\minikube.exe"
)
foreach ($candidate in $MinikubeCandidates) {
    if (Test-Path $candidate) {
        $env:PATH = "$(Split-Path $candidate);$env:PATH"
        break
    }
}
if (-not (Get-Command minikube -ErrorAction SilentlyContinue)) {
    Write-Error "minikube не найден. Перезапустите PowerShell или: winget install Kubernetes.minikube"
    exit 1
}

Write-Host "Starting Minikube..."
minikube start

Write-Host "Enabling ingress addon..."
minikube addons enable ingress

Write-Host "Building Docker images inside Minikube..."
if (-not (Test-Path "$Root\data\raw\geo-reviews-dataset-2023.csv")) {
    Write-Warning "data\raw\geo-reviews-dataset-2023.csv не найден — retrain в K8s не сработает. Выполните: dvc pull"
}
if (-not (Test-Path "$Root\models\model.pkl")) {
    Write-Warning "models\model.pkl не найден — API будет без локальной модели (нужен MLflow в образе)."
}
if (-not (Test-Path "$Root\mlruns\models")) {
    Write-Warning "mlruns\ не найден — MLflow в K8s будет пустым. Скопируйте mlruns от Полины."
}
minikube docker-env | Invoke-Expression
docker build -f "$Root\docker\Dockerfile" -t mlops-api:latest "$Root"
docker build -f "$Root\docker\Dockerfile.mlflow.k8s" -t mlops-mlflow:latest "$Root"

Write-Host "Applying Kubernetes manifests..."
kubectl apply -k "$Root\k8s"

Write-Host "Waiting for API deployment..."
kubectl rollout status deployment/sentiment-api -n mlops-sentiment --timeout=180s

Write-Host ""
Write-Host "Done! Get service URL:"
Write-Host "  minikube service sentiment-api -n mlops-sentiment --url"
Write-Host "Or NodePort: http://$(minikube ip):30080"
