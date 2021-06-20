
if [ $# -eq 0 ]
then
	echo "Please specified the argument [flag]. If your system has installed Anaconda software, please set [flag] to 1, otherwise set [flag] to 0. Then run: sh install.sh [flag]"
	exit 8
fi
flag=$1  # the indicator to decide installing anaconda or not.

# re-download a large trained GBT file
if [ ! -f "trainedmodels/gbt-s4169.pkl" ];
then
	wget https://media.githubusercontent.com/media/Liuxg16/largefiles/8167d5c365c92d08a81dffceff364f72d765805c/gbt-s4169.pkl -P trainedmodels/
fi

# install anaconda
if [ ${flag} = 0 ] ; then
if [ -d ~/anaconda3 ]
then
	echo "Warning: Directory ~/anaconda3 is detected, we are skipping the installation of Anaconda3. More information, please refer to the GeoPPI documentation."
else
	echo "Installing Anaconda..."
	wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
	sh Anaconda3-2020.11-Linux-x86_64.sh -b
fi
fi


eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# build environment
export PATH=$PATH:~/anaconda3/bin/
source activate
conda create -n ppi python==3.8.5 -y
conda activate ppi

# dependencies
pip install  --no-cache-dir torch==1.7.0+cpu -f  https://download.pytorch.org/whl/torch_stable.html
pip install  --no-cache-dir torch==1.7.0+cpu -f  https://download.pytorch.org/whl/torch_stable.html
pip install  --no-cache-dir torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html  
pip install  --no-cache-dir  torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install  --no-cache-dir  torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install  --no-cache-dir  torch-geometric==1.4.1
pip install  --no-cache-dir  scikit-learn==0.24.1
conda install -c schrodinger pymol -y
