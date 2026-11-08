@echo off
rem NeuroShaqma 一键体验：双击即可（自动定位本目录 + 自动装依赖，不用 cd）
rem 若 python 不在 PATH，把下面的 python 改成你电脑上 python.exe 的完整路径
chcp 65001 >nul
echo [1/2] 检查依赖 numpy ...
python -c "import numpy" 2>nul
if errorlevel 1 (
    echo 未安装 numpy，正在自动安装 ...
    pip install -r "%~dp0requirements.txt"
)
echo [2/2] 开始一分钟体验：
echo.
python -X utf8 "%~dp0demo.py"
pause
