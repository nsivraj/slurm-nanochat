#!/bin/bash
# Environment Setup Script for nanochat on Ptolemy HPC
# Run this at the start of each session

echo "=================================================="
echo "nanochat - Environment Setup for Ptolemy HPC"
echo "=================================================="

# Load required modules
echo "Loading Python and CUDA modules..."
module purge
module load python/3.12.5
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
echo "NEXT STEPS (on ptolemy-devel-1 or ptolemy-devel-2):"
echo ""
echo "1. Install dependencies:"
echo "   pip install uv"
echo "   pip install -e '.[gpu]'"
echo ""
echo "2. Download training data (REQUIRED - ~30-60 min):"
echo "   bash scripts/download_data.sh"
echo ""
echo "3. (Optional) Test GPU:"
echo "   python scripts/test_gpu.py"
echo ""
echo "4. Submit SLURM job (from any login node):"
echo "   sbatch scripts/speedrun.slurm"
echo "=================================================="
