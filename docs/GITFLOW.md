# Git Flow и Conventional Commits

## Git Flow

Используем упрощённый **Git Flow**:

| Ветка | Назначение |
|-------|------------|
| `main` | Production-ready код, деплой в K8s |
| `develop` | Интеграционная ветка для фич |
| `feature/*` | Новые фичи → merge в `develop` |
| `fix/*` | Багфиксы → merge в `develop` |
| `release/*` | Подготовка релиза → merge в `main` и `develop` |

### Типичный workflow

```bash
git checkout develop
git pull
git checkout -b feature/drift-monitoring
# ... commits ...
git push -u origin feature/drift-monitoring
# Pull Request → develop

# После тестов:
# PR develop → main → CI/CD deploy
```

## Conventional Commits

Формат сообщения:

```
<type>(<scope>): <subject>

[optional body]
```

### Типы

| type | Когда |
|------|-------|
| `feat` | Новая функциональность |
| `fix` | Исправление бага |
| `docs` | Документация |
| `refactor` | Рефакторинг без изменения поведения |
| `test` | Тесты |
| `ci` | CI/CD |
| `chore` | Прочее (зависимости, конфиги) |

### Примеры

```
feat(api): add drift check endpoint
fix(model): load v4 custom estimator in Docker
docs(readme): add minikube deployment guide
ci: push docker image to GHCR on main
test(drift): add prometheus metrics test
```

### Проверка локально

```bash
# Сообщение должно начинаться с type:
git log --oneline -5
```
