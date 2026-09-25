$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================================"
Write-Host " MIMIR OPTIMAL V8 - SETUP"
Write-Host "============================================================"

if (-not (Test-Path ".env")) {
    Write-Host "[INFO] .env yok. .env.example dosyasını .env olarak kopyalayıp keylerini ekle."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[1/4] .venv oluşturuluyor..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.14 -m venv .venv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path ".venv\Scripts\python.exe")) {
            & py -3 -m venv .venv
        }
    }
    else {
        & python -m venv .venv
    }
}
else {
    Write-Host "[1/4] .venv mevcut."
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "[2/4] Python paketleri kuruluyor/güncelleniyor..."
& $Python -m pip install -U pip
& $Python -m pip install -r requirements.txt

Write-Host "[3/4] ffmpeg / ffprobe / curl kontrolü..."
foreach ($Tool in @("ffmpeg", "ffprobe", "curl")) {
    if (-not (Get-Command $Tool -ErrorAction SilentlyContinue)) {
        throw "$Tool bulunamadı. PATH'e ekle ve setup'ı tekrar çalıştır."
    }
    Write-Host "[OK] $Tool"
}

Write-Host "[4/4] Kod doğrulanıyor..."
& $Python -m compileall -q ai main.py verify_optimal_v8.py
if ($LASTEXITCODE -ne 0) { throw "compileall başarısız." }

& $Python .\verify_optimal_v8.py
if ($LASTEXITCODE -ne 0) { throw "Optimal V8 doğrulaması başarısız." }

Write-Host ""
Write-Host "[DONE] MIMIR Optimal V8 hazır."
Write-Host "Model kontrolü: .\.venv\Scripts\python.exe -m ai.model_check"
Write-Host "İlk test:      .\.venv\Scripts\python.exe -m ai.shorts_pipeline `"C:\path\vod.mp4`" --force --keep-temp"
