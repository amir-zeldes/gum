@REM
@REM Copyright 2009 Humboldt-Universität zu Berlin, INRIA.
@REM
@REM Licensed under the Apache License, Version 2.0 (the "License");
@REM you may not use this file except in compliance with the License.
@REM You may obtain a copy of the License at
@REM
@REM       http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS,
@REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM See the License for the specific language governing permissions and
@REM limitations under the License.
@REM
@REM
@REM
@REM mode con:cols=121
java -Xmx4096m -cp lib/*;plugins/*; -Dfile.encoding=UTF-8 -Dlogback.configurationFile=./conf/logback.xml org.corpus_tools.pepper.cli.PepperStarter %1 %2
