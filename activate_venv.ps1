# PowerShell 가상환경 활성화 스크립트
# 사용법: .\activate_venv.ps1
#
# 이 스크립트는 다음 통상적인 명령을 자동으로 실행합니다:
# .\.venv\Scripts\Activate.ps1

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "가상환경 활성화 중..." -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. 실행 정책 확인
$currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
Write-Host "`n[1] 현재 실행 정책: $currentPolicy" -ForegroundColor White

if ($currentPolicy -eq "Restricted" -or $currentPolicy -eq "Undefined") {
    Write-Host "    실행 정책을 RemoteSigned로 변경합니다..." -ForegroundColor Yellow
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        Write-Host "    ✓ 실행 정책 변경 완료" -ForegroundColor Green
    } catch {
        Write-Host "    ⚠ 실행 정책 변경 실패 (관리자 권한 필요 가능)" -ForegroundColor Yellow
    }
}

# 2. .venv 폴더 확인
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "`n[2] .venv\Scripts\Activate.ps1 발견" -ForegroundColor Green
} else {
    Write-Host "`n[2] 오류: .venv\Scripts\Activate.ps1 파일이 없습니다!" -ForegroundColor Red
    Write-Host "    다음 명령으로 가상환경을 생성하세요:" -ForegroundColor Yellow
    Write-Host "    python -m venv .venv" -ForegroundColor Cyan
    exit 1
}

# 3. 통상적인 방법으로 가상환경 활성화
Write-Host "`n[3] 가상환경 활성화 실행 중..." -ForegroundColor Yellow
Write-Host "    명령: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan

# 직접 dot-sourcing으로 실행 (스크립트를 현재 세션에 로드)
. ".\.venv\Scripts\Activate.ps1"

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host "✓ 가상환경 활성화 완료!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "`n프롬프트 앞에 (.venv)가 표시되어야 합니다." -ForegroundColor Yellow
Write-Host "`nPython 경로 확인: (Get-Command python).Source" -ForegroundColor Cyan
Write-Host "비활성화: deactivate" -ForegroundColor Cyan
