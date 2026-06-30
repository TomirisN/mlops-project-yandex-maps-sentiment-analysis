# Установка Argo CD в Minikube (Windows)
# Запуск: .\scripts\install_argocd.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$MinikubeCandidates = @(
    "$env:USERPROFILE\.bin\minikube.exe",
    "$env:ProgramFiles\Kubernetes\Minikube\minikube.exe"
)
foreach ($candidate in $MinikubeCandidates) {
    if (Test-Path $candidate) {
        $env:PATH = "$(Split-Path $candidate);$env:PATH"
        break
    }
}

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Error "kubectl не найден"
    exit 1
}

Write-Host "Установка Argo CD..."
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

Write-Host "Ожидание готовности Argo CD..."
kubectl rollout status deployment/argocd-server -n argocd --timeout=300s

Write-Host "Применение Application manifest..."
kubectl apply -f "$Root\k8s\argocd-application.yaml"

Write-Host ""
Write-Host "Argo CD UI (в отдельном терминале):"
Write-Host "  kubectl port-forward svc/argocd-server -n argocd 8080:443"
Write-Host "  https://localhost:8080  (логин: admin)"
Write-Host ""
Write-Host "Пароль admin:"
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | ForEach-Object {
    [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_))
}
Write-Host ""
Write-Host "В UI: Applications -> mlops-sentiment -> Sync"
