@rem The CPU SIMD kernels run on jdk.incubator.vector, which Java 17 and later runtimes carry unless their image was
@rem linked without it. For a command that runs the engine in this JVM the module is resolved at start-up, so the
@rem launcher need not relaunch itself to add it: when the runtime's release file (%JAVA_HOME%\release, or the one
@rem two directories above the first java.exe on the PATH) names Java 17 or later and, where it lists the image's
@rem modules, names jdk.incubator.vector among them. The other commands (serve, run-batch, chat, complete, version,
@rem help, cache-list, collect-env, --list-devices) never load the CPU kernels here and start without it.
set "NEURALFREE_VECTOR_COMMAND=1"
for %%c in ("" serve --serve server --server run-batch --run-batch chat --chat complete --complete version --version help --help -h cache-list --cache-list -cl collect-env --collect-env --list-devices) do if "%~1"=="%%~c" set "NEURALFREE_VECTOR_COMMAND="
if not defined NEURALFREE_VECTOR_COMMAND goto neuralfreeVectorModuleDone
set "NEURALFREE_JAVA_RELEASE_FILE="
if defined JAVA_HOME set "NEURALFREE_JAVA_RELEASE_FILE=%JAVA_HOME%\release"
if not defined JAVA_HOME for /f "delims=" %%p in ('where java.exe 2^>NUL') do if not defined NEURALFREE_JAVA_RELEASE_FILE set "NEURALFREE_JAVA_RELEASE_FILE=%%~dpp..\release"
if not defined NEURALFREE_JAVA_RELEASE_FILE goto neuralfreeVectorModuleDone
if not exist "%NEURALFREE_JAVA_RELEASE_FILE%" goto neuralfreeVectorModuleDone
set "NEURALFREE_JAVA_VERSION="
for /f "tokens=1* delims==" %%a in ('findstr /b /c:"JAVA_VERSION=" "%NEURALFREE_JAVA_RELEASE_FILE%"') do set "NEURALFREE_JAVA_VERSION=%%~b"
set NEURALFREE_JAVA_FEATURE=0
if defined NEURALFREE_JAVA_VERSION for /f "tokens=1 delims=.-+_" %%v in ("%NEURALFREE_JAVA_VERSION%") do set "NEURALFREE_JAVA_FEATURE=%%v"
set /a NEURALFREE_JAVA_FEATURE=%NEURALFREE_JAVA_FEATURE% 2>NUL || set NEURALFREE_JAVA_FEATURE=0
if %NEURALFREE_JAVA_FEATURE% LSS 17 goto neuralfreeVectorModuleDone
set "NEURALFREE_JAVA_MODULES="
for /f "tokens=1* delims==" %%a in ('findstr /b /c:"MODULES=" "%NEURALFREE_JAVA_RELEASE_FILE%"') do set "NEURALFREE_JAVA_MODULES=%%~b"
if not defined NEURALFREE_JAVA_MODULES goto neuralfreeVectorModuleAdd
for %%m in (%NEURALFREE_JAVA_MODULES%) do if "%%m"=="jdk.incubator.vector" goto neuralfreeVectorModuleAdd
goto neuralfreeVectorModuleDone
:neuralfreeVectorModuleAdd
set DEFAULT_JVM_OPTS="--add-modules=jdk.incubator.vector" %DEFAULT_JVM_OPTS%
:neuralfreeVectorModuleDone
