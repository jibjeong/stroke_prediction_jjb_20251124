# 가상환경 활성화 상태 확인 스크립트
# 사용법: .\check_venv.ps1

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "가상환경 상태 확인" -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. 환경 변수 VIRTUAL_ENV 확인
Write-Host "`n[1] VIRTUAL_ENV 환경 변수:" -ForegroundColor White
if ($env:VIRTUAL_ENV) {
    Write-Host "    $env:VIRTUAL_ENV" -ForegroundColor Green
    Write-Host "    ✓ 가상환경이 활성화되어 있습니다." -ForegroundColor Green
} else {
    Write-Host "    (설정되지 않음)" -ForegroundColor Red
    Write-Host "    ✗ 가상환경이 활성화되지 않았습니다." -ForegroundColor Red
}

# 2. Python 실행 파일 경로
Write-Host "`n[2] Python 실행 파일 경로:" -ForegroundColor White
try {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
    Write-Host "    $pythonPath" -ForegroundColor Cyan

    if ($pythonPath -like "*\.venv\*") {
        Write-Host "    ✓ .venv 가상환경의 Python을 사용 중입니다." -ForegroundColor Green
    } elseif ($pythonPath -like "*\stroke_prediction_jjb_20251124\*") {
        Write-Host "    ⚠ 프로젝트 폴더 내 Python이지만 .venv가 아닙니다." -ForegroundColor Yellow
    } else {
        Write-Host "    ✗ 시스템 Python을 사용 중입니다." -ForegroundColor Red
    }
} catch {
    Write-Host "    (Python을 찾을 수 없습니다)" -ForegroundColor Red
}

# 3. pip 실행 파일 경로
Write-Host "`n[3] pip 실행 파일 경로:" -ForegroundColor White
try {
    $pipPath = (Get-Command pip -ErrorAction Stop).Source
    Write-Host "    $pipPath" -ForegroundColor Cyan
} catch {
    Write-Host "    (pip를 찾을 수 없습니다)" -ForegroundColor Red
}

# 4. 설치된 패키지 확인
Write-Host "`n[4] 주요 패키지 설치 여부:" -ForegroundColor White
try {
    $packages = pip list 2>$null | Select-String -Pattern "pandas|numpy|scikit-learn|xgboost"
    if ($packages) {
        Write-Host "    ✓ 프로젝트 패키지가 설치되어 있습니다:" -ForegroundColor Green
        $packages | ForEach-Object { Write-Host "      $_" -ForegroundColor Cyan }
    } else {
        Write-Host "    ⚠ 프로젝트 패키지가 설치되지 않았습니다." -ForegroundColor Yellow
    }
} catch {
    Write-Host "    (확인 실패)" -ForegroundColor Red
}

# 5. 프롬프트 표시
Write-Host "`n[5] 현재 프롬프트:" -ForegroundColor White
$promptText = $env:VIRTUAL_ENV_PROMPT
if ($promptText) {
    Write-Host "    $promptText" -ForegroundColor Cyan
} else {
    Write-Host "    (VIRTUAL_ENV_PROMPT 설정되지 않음)" -ForegroundColor Yellow
}

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host "요약" -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan

if ($env:VIRTUAL_ENV -and ($env:VIRTUAL_ENV -like "*\.venv*")) {
    Write-Host "`n✓ .venv 가상환경이 정상적으로 활성화되어 있습니다!" -ForegroundColor Green
    Write-Host "`n예상 프롬프트: (.venv) PS ..." -ForegroundColor Cyan
} elseif ($env:VIRTUAL_ENV) {
    Write-Host "`n⚠ 가상환경은 활성화되어 있지만 .venv가 아닙니다." -ForegroundColor Yellow
    Write-Host "  활성화된 환경: $env:VIRTUAL_ENV" -ForegroundColor Cyan
    Write-Host "`n.venv를 사용하려면:" -ForegroundColor Yellow
    Write-Host "  1. deactivate" -ForegroundColor Cyan
    Write-Host "  2. .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
} else {
    Write-Host "`n✗ 가상환경이 활성화되지 않았습니다." -ForegroundColor Red
    Write-Host "`n활성화 방법:" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "  또는" -ForegroundColor Yellow
    Write-Host "  .\activate_venv.ps1" -ForegroundColor Cyan
}

Write-Host "`n==================================================================" -ForegroundColor Cyan
