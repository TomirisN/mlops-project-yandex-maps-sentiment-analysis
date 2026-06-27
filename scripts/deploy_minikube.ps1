# Deploy to Minikube (Windows PowerShell)
# Usage: .\scripts\deploy_minikube.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "Starting Minikube..."
minikube start

Write-Host "Enabling ingress addon..."
minikube addons enable ingress

Write-Host "Building Docker images inside Minikube..."
minikube docker-env | Invoke-Expression
docker build -f "$Root\docker\Dockerfile" -t mlops-api:latest "$Root"
docker build -f "$Root\docker\Dockerfile.mlflow" -t mlops-mlflow:latest "$Root\docker"

Write-Host "Applying Kubernetes manifests..."
kubectl apply -k "$Root\k8s"

Write-Host "Waiting for API deployment..."
kubectl rollout status deployment/sentiment-api -n mlops-sentiment --timeout=180s

Write-Host ""
Write-Host "Done! Get service URL:"
Write-Host "  minikube service sentiment-api -n mlops-sentiment --url"
Write-Host "Or NodePort: http://$(minikube ip):30080"
