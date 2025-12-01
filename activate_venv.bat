@echo off
REM 가상환경 활성화 배치 파일
REM 사용법: activate_venv.bat
REM
REM 이 스크립트는 다음 통상적인 명령을 자동으로 실행합니다:
REM .venv\Scripts\activate.bat

echo ==================================================================
echo 가상환경 활성화 중...
echo ==================================================================

REM .venv\Scripts\activate.bat 확인
if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo [오류] .venv\Scripts\activate.bat 파일이 없습니다!
    echo 다음 명령으로 가상환경을 생성하세요:
    echo     python -m venv .venv
    echo.
    pause
    exit /b 1
)

echo.
echo [실행] .venv\Scripts\activate.bat
echo.

REM 통상적인 방법으로 가상환경 활성화
call .venv\Scripts\activate.bat

echo.
echo ==================================================================
echo 가상환경 활성화 완료!
echo ==================================================================
echo.
echo 프롬프트 앞에 (.venv)가 표시되어야 합니다.
echo.
echo Python 경로 확인: where python
echo 비활성화: deactivate
echo ==================================================================
