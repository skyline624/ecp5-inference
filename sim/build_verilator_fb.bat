@echo off
REM build_verilator_fb.bat -- compile le C++ Verilator (vobj/*.cpp) vers un exe MSVC.
REM Top = gen_seq_top (DUT direct), TB = sim_main.cpp (C++, untimed, drive clk).
REM Lance depuis sim/ apres : verilator --cc ... --top-module gen_seq_top -Mdir vobj ...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64 FAILED & exit /b 1 )
set VINC=C:\oss-cad-suite\share\verilator\include
cl /nologo /EHsc /O2 /std:c++20 /permissive- /W1 ^
   /I vobj /I "%VINC%" /I "%VINC%\vltstd" ^
   /Fe:vobj\tb_verilator.exe /Fo:vobj\ ^
   vobj\*.cpp sim_main.cpp ^
   "%VINC%\verilated.cpp" "%VINC%\verilated_timing.cpp" "%VINC%\verilated_threads.cpp" 2>&1
echo CL_EXIT=%errorlevel%