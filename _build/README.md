# Setting up a conda environment and compiling dependencies for the neural constituency parser

#### Step One
To setup a conda environment named 'gum' to host the build process and install the python dependencies for the build bot, run the following command from the root (gum) folder: 
> conda env create --name gum --file environment.yml <br/>
> conda activate gum

#### Step Two
Step Two (*Building the Cython dependencies*) is optional and needs to be done only if you intend to re-generate the constituent parses using the neural parser with the -p flag. <br/> <br/>
Run the following to compile the cython packages used by the neural parser (assuming you're still at the root (gum) folder):
 
Navigate to the _build folder:
> cd _build

(*Windows machines only*) Install the Visual C++ Build Tools, needed for compiling the Cython modules. This is a one-time installation activity on the Host OS.
> Download and Install the latest Build Tools for Visual Studio 2019 from [here](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) <br />
> For other versions, please check [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Then run this to compile and install the cython packages. 
>  python setup.py build_ext --inplace

If this works, you should see these files in the _build folder (indicating successful compilation). If no, consider upgrading cython. 

> const_decoder.c <br/> 
> hpsg_decoder.c <br/>
> const_decoder.cpython-37m-x86_64-linux-gnu.so (on Ubuntu / Linux) <br/>
> const_decoder.cp37-win_amd64 (on Windows 64-bit) <br/>
> const_decoder.cpython-37m-darwin.so (on MacOS) <br/>
> hpsg_decoder.cpython-37m-x86_64-linux-gnu.so (on Ubuntu / Linux) <br/>
> hspg_decoder.cp37-win_amd64 (on Windows 64-bit) <br/>
> hpsg_decoder.cpython-37m-darwin.so (on MacOS) <br/>

#### Step Three
At this stage, the build bot process can be started as normal<br /> 
Run the following to start the build bot. and generate fresh constituent parses 

> python build_gum.py -s src -t target -p -v

This will download the best PyTorch model for constituent parsing automatically and summon it for inferring the constituent parses, which are saved to the folder target/const 
