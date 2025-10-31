#!/bin/bash
# Environment Setup Script for nanochat on Ptolemy HPC
# Run this at the start of each session

echo "=================================================="
echo "nanochat - Environment Setup for Ptolemy HPC"
echo "=================================================="

# Load required modules
echo "Loading CUDA modules..."
module purge
module load cuda-toolkit
module load spack-managed-x86-64_v3/v1.0

# Show loaded modules
echo -e "\nLoaded modules:"
module list

# Set up Python virtual environment on scratch
VENV_PATH="/scratch/ptolemy/users/$USER/nanochat-venv"

if [ ! -d "$VENV_PATH" ]; then
    echo -e "\nCreating virtual environment at $VENV_PATH..."
    python3 -m venv $VENV_PATH
else
    echo -e "\nVirtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_PATH/bin/activate

# Show Python and pip versions
echo -e "\nPython version: $(python --version)"
echo "Pip version: $(pip --version)"

# Set environment variables for nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="/scratch/ptolemy/users/$USER/cache/huggingface"
export TORCH_HOME="/scratch/ptolemy/users/$USER/cache/torch"
export CARGO_HOME="/scratch/ptolemy/users/$USER/.cargo"
export RUSTUP_HOME="/scratch/ptolemy/users/$USER/.rustup"
export UV_CACHE_DIR="/scratch/ptolemy/users/$USER/cache/uv"
export PIP_CACHE_DIR="/scratch/ptolemy/users/$USER/cache/pip"

# Create cache directories if they don't exist
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p $CARGO_HOME
mkdir -p $RUSTUP_HOME
mkdir -p $UV_CACHE_DIR
mkdir -p $PIP_CACHE_DIR

echo -e "\nEnvironment variables set:"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "HF_HOME: $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"
echo "CARGO_HOME: $CARGO_HOME"
echo "RUSTUP_HOME: $RUSTUP_HOME"
echo "UV_CACHE_DIR: $UV_CACHE_DIR"
echo "PIP_CACHE_DIR: $PIP_CACHE_DIR"

# Test CUDA availability (if PyTorch is installed)
echo -e "\nTesting CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not installed yet"

echo -e "\n=================================================="
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Install dependencies:"
echo "   ssh to ptolemy-devel-1.arc.msstate.edu"
echo "   source scripts/setup_environment.sh"
echo "   pip install uv"
echo "   uv sync --extra gpu"
echo ""
echo "2. Submit SLURM job:"
echo "   sbatch --mail-user=your_email@msstate.edu scripts/speedrun.slurm"
echo "=================================================="
