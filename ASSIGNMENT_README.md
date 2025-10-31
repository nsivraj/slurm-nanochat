# nanochat for CSE8990 Assignment 3

This repository contains the nanochat codebase configured to run on Ptolemy HPC for Assignment 3.

## Assignment Context

**Assignment:** Transformer Architecture Analysis
**Due Date:** October 31, 2025
**Objective:** Study the nanochat codebase and explain how it implements the Transformer architecture

## Repository Structure

This is a fork/clone of [karpathy/nanochat](https://github.com/karpathy/nanochat) with Ptolemy HPC-specific configuration added:

### New Files for Ptolemy HPC

- `scripts/setup_environment.sh` - Environment setup script for Ptolemy
- `scripts/speedrun.slurm` - SLURM job script for 8xA100 training
- `scripts/test_gpu.py` - GPU configuration test script
- `.env.local.example` - Example configuration file
- `PTOLEMY_SETUP.md` - Detailed setup and usage guide
- `ASSIGNMENT_README.md` - This file

### Original nanochat Files

All other files are from the original nanochat repository. Key files for the assignment:

**Core Model Implementation:**
- `nanochat/gpt.py` - The GPT Transformer architecture
- `nanochat/dataloader.py` - Tokenizing distributed data loader
- `nanochat/tokenizer.py` - BPE tokenizer wrapper
- `nanochat/engine.py` - Model inference with KV cache

**Training Scripts:**
- `scripts/base_train.py` - Base model pretraining
- `scripts/mid_train.py` - Midtraining (conversation format)
- `scripts/chat_sft.py` - Supervised finetuning

**Evaluation:**
- `nanochat/core_eval.py` - CORE score evaluation
- `scripts/chat_eval.py` - Chat model evaluation

## Quick Start for Assignment

### 1. Initial Setup (One-time, on ptolemy-devel-1)

See [PTOLEMY_SETUP.md](PTOLEMY_SETUP.md) for detailed instructions.

```bash
# SSH to devel node (has internet access)
ssh [username]@ptolemy-devel-1.arc.msstate.edu

# Navigate to scratch
cd /scratch/ptolemy/users/$USER
git clone [your-repo-url] slurm-nanochat
cd slurm-nanochat

# Configure email
cp .env.local.example .env.local
nano .env.local  # Set your email

# Setup environment
bash scripts/setup_environment.sh
pip install uv
uv sync --extra gpu
```

### 2. Download Training Data (REQUIRED, on ptolemy-devel-1)

**CRITICAL:** GPU compute nodes do NOT have internet access!

```bash
# Still on ptolemy-devel-1
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Download all required data (~30-60 minutes for ~24GB)
bash scripts/download_data.sh
```

This downloads:
- 240 dataset shards (~24GB)
- Evaluation bundle
- Identity conversations
- Builds and trains the tokenizer

### 3. Submit Training Job (any Ptolemy node)

```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Create logs directory
mkdir -p logs

# Submit job
sbatch scripts/speedrun.slurm
```

The job will verify all data is downloaded before starting training.

### 4. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch output
tail -f logs/nanochat_speedrun_*.out
```

## Assignment Deliverables

The assignment requires a written report analyzing:

### 1. Code Analysis
Study these key files:
- `nanochat/gpt.py` - Main Transformer implementation
- `nanochat/dataloader.py` - Data processing
- `nanochat/tokenizer.py` - Tokenization
- `scripts/base_train.py` - Training loop

### 2. Transformer Components Mapping

Identify in the code:
- **Token and positional embeddings** → `nanochat/gpt.py:GPT.__init__`, `wte`, `wpe`
- **Multi-head self-attention** → `nanochat/gpt.py:CausalSelfAttention`
- **Feedforward layers** → `nanochat/gpt.py:MLP`
- **Layer normalization** → `nanochat/gpt.py:Block` (ln_1, ln_2)
- **Residual connections** → `nanochat/gpt.py:Block.forward`
- **Causal masking** → `nanochat/gpt.py:CausalSelfAttention`
- **Output projection** → `nanochat/gpt.py:GPT.forward` (lm_head)

### 3. Forward Pass Trace

Trace the data flow through:
1. Input tokens → `tokenizer.encode()`
2. Token embeddings → `self.wte(idx)`
3. Positional embeddings → `self.wpe(pos)`
4. Transformer blocks → `for block in self.blocks`
5. Output projection → `self.lm_head(x)`
6. Softmax → `F.softmax(logits, dim=-1)`

### 4. Training Process

Examine:
- Loss calculation → `scripts/base_train.py` (cross-entropy)
- Optimizer → `nanochat/muon.py` or `nanochat/adamw.py`
- Gradient updates → `scripts/base_train.py:main()`
- Evaluation metrics → `nanochat/core_eval.py`

## Using the Trained Model

After training completes (~4-5 hours), you can:

### View Training Report

```bash
cat report.md
```

This contains:
- Model configuration
- Training metrics
- Evaluation scores (CORE, ARC, GSM8K, HumanEval, MMLU)
- Sample outputs

### Chat with the Model

```bash
# Request interactive GPU session
srun --account=class-cse8990 --partition=gpu-a100 --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash

# Activate environment
cd /scratch/ptolemy/users/$USER/nanochat
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"

# Chat via CLI
python -m scripts.chat_cli -p "Explain how transformers work"
```

## Code Exploration Tips

### Understanding the Architecture

Start with `nanochat/gpt.py`:

```python
# Read the GPT class
class GPT(nn.Module):
    def __init__(self, config):
        # Token embedding table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embedding table
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # Output layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

### Understanding the Training Loop

Study `scripts/base_train.py`:

```python
# Training loop structure:
for step in range(num_steps):
    # 1. Get batch of data
    tokens = loader.next_batch()
    # 2. Forward pass
    logits, loss = model(tokens)
    # 3. Backward pass
    loss.backward()
    # 4. Optimizer step
    optimizer.step()
    # 5. Evaluate periodically
    if step % eval_interval == 0:
        evaluate(model)
```

### Tracing Multi-Head Attention

In `nanochat/gpt.py:CausalSelfAttention`:

```python
# Q, K, V projections
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
# Reshape for multi-head
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# Scaled dot-product attention
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# Causal mask
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
# Softmax + dropout
att = F.softmax(att, dim=-1)
att = self.attn_dropout(att)
# Apply attention to values
y = att @ v
```

## Expected Training Results

On 8xA100 GPUs, the d20 speedrun model (561M parameters) should achieve:

- **Training Time:** ~4 hours
- **CORE Score:** ~0.22
- **ARC-Easy:** ~0.35-0.40
- **GSM8K:** ~0.02-0.05
- **Model Size:** 561M parameters
- **Training Data:** ~11.2B tokens

These numbers will be in your `report.md` after training.

## Resources

- **Original nanochat:** [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
- **Ptolemy HPC Docs:** [hpc.msstate.edu](https://www.hpc.msstate.edu/computing/ptolemy/)
- **Setup Guide:** [PTOLEMY_SETUP.md](PTOLEMY_SETUP.md)
- **Main README:** [README.md](README.md)

## Troubleshooting

See [PTOLEMY_SETUP.md](PTOLEMY_SETUP.md) for detailed troubleshooting.

Common issues:
- **OOM errors:** Reduce `--device_batch_size` in SLURM script
- **Dependency issues:** Install on `ptolemy-devel-1` server
- **Slow downloads:** Pre-download dataset in interactive session

## Assignment Timeline

1. **Week 1:** Setup environment and submit training job
2. **Week 2:** Monitor training, study code while waiting
3. **Week 3:** Analyze results, trace forward pass, write report
4. **Week 4:** Complete report, proofread, submit

## Notes

- Training can run while you work on other parts of the assignment
- The SLURM job is fire-and-forget - it runs unattended
- Email notifications will alert you when training starts/ends/fails
- All outputs are saved to `/scratch/ptolemy/users/$USER/nanochat-cache/`

Good luck with your assignment!
