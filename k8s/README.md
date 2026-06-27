# Kubernetes / Minikube / Argo CD

## Состав

| Файл | Назначение |
|------|------------|
| `namespace.yaml` | Namespace `mlops-sentiment` |
| `configmap.yaml` | Env для API |
| `api-deployment.yaml` / `api-service.yaml` | FastAPI |
| `mlflow-deployment.yaml` | MLflow server |
| `prometheus-deployment.yaml` | Prometheus |
| `grafana-deployment.yaml` | Grafana |
| `ingress.yaml` | Ingress (sentiment.local) |
| `argocd-application.yaml` | GitOps через Argo CD |
| `kustomization.yaml` | `kubectl apply -k k8s/` |

## Minikube (локальный деплой)

```powershell
# из корня репозитория
.\scripts\deploy_minikube.ps1
```

Или вручную:

```powershell
minikube start
minikube addons enable ingress
kubectl apply -k k8s/
minikube service sentiment-api -n mlops-sentiment --url
```

## Argo CD (GitOps)

1. Установите Argo CD в кластер
2. Замените `repoURL` в `argocd-application.yaml` на URL вашего GitHub-репозитория
3. `kubectl apply -f k8s/argocd-application.yaml`
4. Argo CD автоматически синхронизирует `k8s/` при push в `main`

## CI/CD deploy

При push в `main` GitHub Actions:
- публикует образы в GHCR
- деплоит в K8s, если задан secret `KUBE_CONFIG` (base64 kubeconfig)
