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

```powershell
.\scripts\install_argocd.ps1
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Откройте https://localhost:8080 → Application `mlops-sentiment` → **Sync**.

Подробнее: [docs/DEMO.md](../docs/DEMO.md)

## CI/CD deploy

При push в `main` GitHub Actions:
- публикует образы в GHCR
- деплоит в K8s, если задан secret `KUBE_CONFIG` (base64 kubeconfig)
