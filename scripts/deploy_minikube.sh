#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

minikube start
minikube addons enable ingress
eval "$(minikube docker-env)"
docker build -f "$ROOT/docker/Dockerfile" -t mlops-api:latest "$ROOT"
docker build -f "$ROOT/docker/Dockerfile.mlflow" -t mlops-mlflow:latest "$ROOT/docker"
kubectl apply -k "$ROOT/k8s"
kubectl rollout status deployment/sentiment-api -n mlops-sentiment --timeout=180s
echo "URL: $(minikube service sentiment-api -n mlops-sentiment --url)"
