# Локальная настройка проекта (Windows PowerShell)
# Запуск: .\scripts\setup_local.ps1
# Полная установка (torch, BERT): .\scripts\setup_local.ps1 -Full

param(
    [switch]$Full,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$VenvPython = Join-Path $Root "geo_reviews_venv\Scripts\python.exe"
$VenvPip = Join-Path $Root "geo_reviews_venv\Scripts\pip.exe"
$VenvPytest = Join-Path $Root "geo_reviews_venv\Scripts\pytest.exe"

function Write-Step([string]$Num, [string]$Text) {
    Write-Host ""
    Write-Host "[$Num] $Text" -ForegroundColor Cyan
}

function Test-ModuleInstalled([string]$ModuleName) {
    & $VenvPython -c "import $ModuleName" 2>$null
    return $LASTEXITCODE -eq 0
}

Write-Host "=== MLOps Sentiment — локальная настройка ===" -ForegroundColor Green
if ($Full) {
    Write-Host "Режим: полная установка (requirements.txt, включая torch)" -ForegroundColor Yellow
} else {
    Write-Host "Режим: быстрый (requirements-ci.txt — API, тесты, MLflow)" -ForegroundColor Yellow
    Write-Host "Для BERT/torch: .\scripts\setup_local.ps1 -Full" -ForegroundColor DarkGray
}

# --- 1. venv ---
Write-Step "1/6" "Проверка Python 3.11 и venv..."

$venvPythonPath = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $venvPythonPath = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
}
if (-not $venvPythonPath) {
    Write-Error "Нужен Python 3.11. Установите: winget install Python.Python.3.11"
    exit 1
}

if (-not (Test-Path "geo_reviews_venv\Scripts\Activate.ps1")) {
    Write-Host "  Создаём geo_reviews_venv..."
    & py -3.11 -m venv geo_reviews_venv
} else {
    $currentVer = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($currentVer -ne "3.11") {
        Write-Host "  venv на Python $currentVer — пересоздаём на 3.11..."
        Remove-Item -Recurse -Force geo_reviews_venv
        & py -3.11 -m venv geo_reviews_venv
    } else {
        Write-Host "  venv уже есть (Python 3.11) — OK"
    }
}

$pyVer = & $VenvPython --version
Write-Host "  Используем: $pyVer ($VenvPython)"

# --- 2. pip install ---
Write-Step "2/6" "Установка зависимостей..."

$reqFile = if ($Full) { "requirements.txt" } else { "requirements-ci.txt" }
$needInstall = $true
if ((Test-ModuleInstalled "pytest") -and (Test-ModuleInstalled "fastapi") -and (Test-ModuleInstalled "evidently")) {
  if (-not $Full) {
    Write-Host "  Основные пакеты уже установлены — пропускаем pip install"
    Write-Host "  (чтобы переустановить: удалите geo_reviews_venv и запустите снова)"
    $needInstall = $false
  }
}

if ($needInstall) {
    Write-Host "  Файл: $reqFile"
    if ($Full) {
        Write-Host "  Это может занять 15-30 минут (скачивается torch) — НЕ зависло, ждите..." -ForegroundColor Yellow
    } else {
        Write-Host "  Обычно 3-10 минут — НЕ зависло, ждите..." -ForegroundColor Yellow
    }
    & $VenvPython -m pip install --upgrade pip --disable-pip-version-check
    & $VenvPip install -r $reqFile --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install завершился с ошибкой"
        exit 1
    }
    Write-Host "  Зависимости установлены" -ForegroundColor Green
}

# --- 3. model.pkl ---
Write-Step "3/6" "Проверка модели..."
New-Item -ItemType Directory -Force -Path models | Out-Null
if ((Test-Path "model.pkl") -and -not (Test-Path "models\model.pkl")) {
    Move-Item -Force model.pkl models\model.pkl
    Write-Host "  Перенесён model.pkl -> models\model.pkl"
} elseif (Test-Path "models\model.pkl") {
    Write-Host "  models\model.pkl — OK"
} else {
    Write-Host "  models\model.pkl не найден (обучите: python src\core\train_mlflow.py)" -ForegroundColor Yellow
}

# --- 4. DVC / датасет ---
Write-Step "4/6" "Проверка датасета..."
$dataset = "data\raw\geo-reviews-dataset-2023.csv"
if (Test-Path $dataset) {
    Write-Host "  Датасет — OK"
} else {
    Write-Host "  Датасет не найден. Пробуем dvc pull..."
    if (Test-ModuleInstalled "dvc") {
        & (Join-Path $Root "geo_reviews_venv\Scripts\dvc.exe") pull
    } else {
        Write-Host "  dvc не установлен (в быстром режиме это нормально)" -ForegroundColor Yellow
    }
    if (-not (Test-Path $dataset)) {
        Write-Host "  Датасет всё ещё отсутствует. Для dvc pull сначала:" -ForegroundColor Yellow
        Write-Host "    cd docker; docker compose up -d minio minio-init; cd ..; dvc pull"
    }
}

# --- 5. reference + mlruns ---
Write-Step "5/6" "Reference sample и mlruns..."
if (Test-Path $dataset) {
    & $VenvPython scripts\init_reference_data.py
    Write-Host "  reference_sample.csv — OK"
} else {
    Write-Host "  Пропуск init_reference_data (нет датасета)" -ForegroundColor Yellow
}
if (-not (Test-Path "mlruns")) {
    New-Item -ItemType Directory -Force -Path mlruns | Out-Null
    Write-Host "  Создана папка mlruns/ (обучите модель или скопируйте от коллеги)"
} else {
    Write-Host "  mlruns/ — OK"
}

# --- 6. тесты ---
Write-Step "6/6" "Запуск тестов..."
if ($SkipTests) {
    Write-Host "  Пропущено (-SkipTests)"
} else {
    & $VenvPytest tests\ -v --tb=short
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Тесты не прошли"
        exit 1
    }
    Write-Host "  Все тесты прошли" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Готово ===" -ForegroundColor Green
Write-Host ""
Write-Host "Активируйте venv (если ещё не активирован):" -ForegroundColor White
Write-Host "  .\geo_reviews_venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Дальше — Docker (рекомендуется для защиты):" -ForegroundColor White
Write-Host "  cd docker" -ForegroundColor White
Write-Host "  docker compose build" -ForegroundColor White
Write-Host "  docker compose up -d" -ForegroundColor White
Write-Host "  Откройте http://127.0.0.1:8000/ui" -ForegroundColor White
Write-Host ""
Write-Host "Подробная инструкция: docs\DEMO.md" -ForegroundColor DarkGray
