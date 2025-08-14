#!/usr/bin/env bash

sudo apt-get update && sudo apt-get install -y libgl1-mesa-dev
eval "$(conda shell.bash hook)"

if [ -d "/work/data_drive/DeepShedTreeMortality" ]; then
    # Check if the github repo is up to date
    cd "/work/data_drive/DeepShedTreeMortality"
    git pull origin treehealth
fi

# Check if dir /work/cenv/ssl4eo exists; create conda env if not
if [ -d "/work/data_drive/cenv/ssl4eo_delfors" ]; then
    export CONDA_ENVS_DIRS="/work/data_drive/cenv/"
    echo  'export CONDA_ENVS_DIRS="/work/data_drive/cenv/"' >> ~/.bashrc
#    mamba env config vars set -n ssl4eo MMEARTH_DIR=/work/data/MMEARTH100K/ GEO_BENCH_DIR=/work/data/geobench
else
    echo "Creating conda environment"
    mamba env create --prefix /work/data_drive/cenv/ssl4eo_delfors -f /work/data_drive/DeepShedTreeMortality/env_cuda12.yml
    pip install -U git+https://github.com/qubvel/segmentation_models.pytorch
    export CONDA_ENVS_DIRS="/work/data_drive/cenv/"
    echo  'export CONDA_ENVS_DIRS="/work/data_drive/cenv/"' >> ~/.bashrc
    # Set environment variables ensuring that env vars are always set when activating env
#    mamba env config vars set -n ssl4eo MMEARTH_DIR=/work/data/MMEARTH100K/ GEO_BENCH_DIR=/work/data/geobench
    # update a script for cuda12

fi
mamba init

# Export new python kernel
/work/data_drive/cenv/ssl4eo_delfors/bin/python -m ipykernel install --user --name ipy39 --display-name "ssl4eo_delfors"
