# Setting up a conda environment and compiling dependencies for the neural constituency parser

To setup a conda environment named 'gum' to host the build process and install the python dependencies for the neural constituency parser, run the following command from the root (gum) folder:

> conda env create --name gum --file environment.yml

> conda activate gum

Next, run the following to compile the cython packages used by the neural constituent parser (assuming you're still at the root (gum) folder):
 
Navigate to the _build folder:
> cd _build

(*Windows machines only*) Install the Visual C++ Build Tools, needed for compiling the Cython modules. This is a one-time setup activity.
> Download and Install the latest Build Tools for Visual Studio 2019 from [here](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)

> For other versions, please check [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Then run this to compile and install the cython packages
>  python setup.py build_ext --inplace

If this works, you should see these files in the _build folder (indicating successful compilation). If no, consider upgrading cython. 
> const_decoder.c 

> hpsg_decoder.c

> const_decoder.cpython-37m-x86_64-linux-gnu.so (on ubuntu build)

> hpsg_decoder.cpython-37m-x86_64-linux-gnu.so (on ubuntu build)

At this stage, the build bot process can be started as normal:
> python build_gum.py -s src -t target -p -v

This will download the best PyTorch model for constituent parsing automatically and summon it for inferring the constituent parses, which are saved to the folder target/const 

Note that the conda environment and cython dependencies must be setup to run the build bot processes, even should you choose to not re-generate the constituent parses. This is a one-time setup activity. 
