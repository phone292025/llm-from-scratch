@echo off
rem Talk to your chat model. Double-click this file, or run it from a terminal:
rem   talk.bat                      newest checkpoint in checkpoints\chat
rem   talk.bat --device cpu         without the GPU (slower, needs ~7 GB of RAM)
rem Any option scripts\talk.py takes works here too; talk.bat --help lists them.
title LLM
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\talk.py %*
rem Keep the window open if something went wrong, so the message can be read.
if errorlevel 1 pause
