@echo off
rem NeuroShaqma 一键体验：双击即可（自动定位本目录，不用 cd）
rem 若 python 不在 PATH，把下面的 python 改成你电脑上 python.exe 的完整路径
chcp 65001 >nul
python -X utf8 "%~dp0demo.py"
pause
