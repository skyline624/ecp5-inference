@echo off
setlocal
cd /d "%~dp0"
set PYDIR=C:\Users\User\AppData\Local\Programs\Python\Python312
set CLIB=%PYDIR%\Lib\site-packages\cocotb\libs
set PATH=%CLIB%;%PYDIR%;%PATH%
set PYGPI_PYTHON_BIN=%PYDIR%\python.exe
set COCOTB_TEST_MODULES=test_gen_F_fb
set TOPLEVEL=gen_seq_top
set TOPLEVEL_LANG=verilog
set PYTHONPATH=%CD%
set COCOTB_LOG_LEVEL=INFO
if "%NTOK%"=="" set NTOK=1
echo === running sim_top.exe (Verilator+cocotb embed, NTOK=%NTOK%) ===
obj_dir\sim_top.exe
echo === exit=%errorlevel% ===
endlocal