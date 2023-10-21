::@echo off
REM Ativar o ambiente virtual
call venv\Scripts\activate.bat
cd "Docs"
make html
pause
