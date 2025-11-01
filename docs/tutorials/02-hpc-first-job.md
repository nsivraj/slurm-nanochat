# Tutorial: Your First HPC Training Job

**Goal:** Submit and complete your first training job on Ptolemy HPC

**What you'll learn:**
- How to set up the Ptolemy environment
- How to download required data
- How to submit a SLURM job
- How to monitor training progress
- How to chat with your trained model

**Prerequisites:**
- Ptolemy HPC account (MSU students)
- SSH access to Ptolemy
- Email address for notifications

**Time:** ~2-3 hours setup, ~12 hours training

---

## Step 1: Initial Setup (One-Time)

### SSH to Development Node

**IMPORTANT:** Use a development node that has internet access!

```bash
ssh <username>@ptolemy-devel-1.arc.msstate.edu
```

### Navigate to Scratch Space

**CRITICAL:** Use `/scratch/ptolemy/users/$USER/` for all storage!

```bash
cd /scratch/ptolemy/users/$USER
```

Home directory has tiny quotas - don't use it!

### Clone the Repository

```bash
git clone <your-repo-url> slurm-nanochat
cd slurm-nanochat
```

### Configure Email Notifications

```bash
cp .env.local.example .env.local
nano .env.local
```

Set your email:
```bash
EMAIL=your_email@msstate.edu
```

Save and exit (Ctrl+X, Y, Enter).

### Run Setup Script

```bash
bash scripts/setup_environment.sh
```

This will:
- Load Python 3.12.5 module
- Create virtual environment in `/scratch/ptolemy/users/$USER/nanochat-venv`
- Install all dependencies

---

## Step 2: Download Training Data

**CRITICAL:** GPU compute nodes have NO internet access!

You MUST download all data on the development node first.

```bash
# Still on ptolemy-devel-1
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Download everything (~30-60 minutes for ~24GB)
bash scripts/download_data.sh
```

This downloads:
- 240 dataset shards (~24GB)
- Evaluation bundle (~162MB)
- Identity conversations (~2.3MB)
- GPT-2/GPT-4 tokenizers (for evaluation)
- SmolTalk dataset (for midtraining)

**Go get coffee - this takes a while!**

---

## Step 3: Submit Your Training Job

### The Critical Command

**IMPORTANT:** You MUST set `WANDB_RUN` when submitting!

```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# ✅ CORRECT - Set WANDB_RUN to any non-"dummy" name
WANDB_RUN=my_first_run sbatch scripts/speedrun.slurm
```

### Why WANDB_RUN Matters

```bash
# ❌ WRONG - Job will fail immediately
sbatch scripts/speedrun.slurm

# ❌ WRONG - Midtraining and SFT will be SKIPPED
WANDB_RUN=dummy sbatch scripts/speedrun.slurm

# ✅ CORRECT - Any non-"dummy" name works
WANDB_RUN=my_run sbatch scripts/speedrun.slurm
WANDB_RUN=ptolemy_training sbatch scripts/speedrun.slurm
WANDB_RUN=test_run_1 sbatch scripts/speedrun.slurm
```

**Note:** You do NOT need a wandb account! The name just can't be "dummy".

### What Happens Next

The job will:
1. Validate WANDB_RUN is set correctly
2. Verify all data is downloaded
3. Request 8xA100 GPUs from SLURM
4. Run the complete training pipeline
5. Send you email notifications

---

## Step 4: Monitor Your Job

### Check Job Status

```bash
squeue -u $USER
```

Output:
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
12345  gpu-a100 nanochat  yourid  R    1:23:45      1 ptolemy-gpu-01
```

- `PD` = Pending (waiting in queue)
- `R` = Running
- No output = Job finished (check logs)

### Watch Training Progress Live

```bash
# Get your job ID from squeue
JOBID=12345  # Replace with your actual job ID

# Watch the output log
tail -f /scratch/ptolemy/users/$USER/slurm-nanochat/logs/nanochat_speedrun_${JOBID}.out
```

Press Ctrl+C to stop watching.

### Check for Errors

```bash
tail -f /scratch/ptolemy/users/$USER/slurm-nanochat/logs/nanochat_speedrun_${JOBID}.err
```

### Email Notifications

You'll receive emails:
- **BEGIN** - Job started
- **END** - Job completed successfully
- **FAIL** - Job failed (check error log)

---

## Step 5: Understanding the Training Timeline

### Full Timeline (~10-12 hours)

```
0:00  → Job starts
0:00  → WANDB_RUN validation passes ✅
0:00  → Data verification passes ✅
0:05  → Tokenizer evaluation complete
7:00  → Base pretraining complete (CORE: ~0.21)
9:30  → Midtraining complete
11:30 → Supervised finetuning complete
11:35 → Report generated
11:36 → Job done! ✅
```

### What Each Phase Does

**Tokenizer Evaluation** (5 min):
- Compares your BPE tokenizer to GPT-2/GPT-4
- Tests compression efficiency

**Base Pretraining** (~7 hours):
- Trains 20-layer, 560M parameter model
- Learns general language patterns
- Evaluated with CORE benchmark

**Midtraining** (~2-3 hours):
- Teaches conversation format
- Adds special tokens
- Enables multi-turn chat

**Supervised Finetuning** (~1-2 hours):
- Improves instruction-following
- Enhances helpfulness
- Final chat model

---

## Step 6: Review Your Results

Once the job completes (check email!):

```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat
cat report.md
```

### Expected Results

You should see:

```markdown
| Metric          | BASE     | MID      | SFT      |
|-----------------|----------|----------|----------|
| CORE            | 0.21     | -        | -        |
| ARC-Challenge   | -        | 0.28     | 0.30     |
| ARC-Easy        | -        | 0.36     | 0.39     |
| GSM8K           | -        | 0.03     | 0.05     |
| HumanEval       | -        | 0.07     | 0.09     |
| MMLU            | -        | 0.31     | 0.32     |
| ChatCORE        | -        | 0.07     | 0.09     |
```

### What the Metrics Mean

- **CORE**: Base model language understanding (~0.21 is good for this size)
- **ARC**: Science questions (Challenge is harder than Easy)
- **GSM8K**: Grade-school math problems
- **HumanEval**: Simple Python coding tasks
- **MMLU**: Broad knowledge across 57 subjects
- **ChatCORE**: Chat-specific language understanding

---

## Step 7: Chat with Your Model

### Request Interactive GPU Session

```bash
srun --account=class-cse8990 --partition=gpu-a100 --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash
```

This gets you a 1-hour interactive session with 1 GPU.

### Activate Environment

```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"
```

### Chat!

```bash
# Single question
python -m scripts.chat_cli -p "Why is the sky blue?"

# Interactive mode
python -m scripts.chat_cli
```

### Try These Prompts

```bash
python -m scripts.chat_cli -p "Explain transformers in simple terms"
python -m scripts.chat_cli -p "Write a Python function to check if a number is prime"
python -m scripts.chat_cli -p "Tell me a joke about programming"
python -m scripts.chat_cli -p "What is 15 * 24?"
```

### Exit Interactive Session

```bash
exit  # Exits the chat
exit  # Exits the GPU session
```

---

## Common Issues and Solutions

### Job Fails Immediately

**Check the error:**
```bash
cat logs/nanochat_speedrun_*.err
```

**Common causes:**
- "WANDB_RUN not set" → Use `WANDB_RUN=my_run sbatch ...`
- "WANDB_RUN is dummy" → Use a different name
- "Data not found" → Re-run `scripts/download_data.sh` on devel node

### Job Times Out at 12 Hours

This is the QOS limit for the class account. Your training should complete within 12 hours. If not:
- Check logs for slow progress
- Base model should complete in ~7 hours

### Chat Doesn't Work

**Error: `FileNotFoundError: chatsft_checkpoints`**

This means SFT didn't run. Use the midtrained model instead:

```bash
python -m scripts.chat_cli -i mid
```

Or re-submit training with correct `WANDB_RUN`.

### Out of Memory (OOM)

Edit `scripts/speedrun.slurm` to reduce batch size:

```bash
--device_batch_size=16  # Reduce from 32
```

---

## What You Built

### Model Specifications

- **Parameters**: 561 million
- **Layers**: 20 transformer blocks
- **Model Dimension**: 1,280
- **Attention Heads**: 10
- **Context Length**: 2,048 tokens
- **Vocabulary**: ~65,536 tokens (BPE)

### Training Scale

- **Training Tokens**: ~11.2 billion
- **Dataset**: FineWeb (web text)
- **Training Time**: ~10-12 hours
- **Hardware**: 8x A100 80GB GPUs
- **Compute Cost**: ~$150 (free with class allocation)

### Performance

Your model:
- ✅ Slightly outperforms GPT-2 (2019)
- ✅ Can chat conversationally
- ✅ Follows basic instructions
- ❌ Still makes mistakes (it's micro-scale!)
- ❌ Not as good as modern LLMs (GPT-5, etc.)

---

## Next Steps

### 1. Download Results Locally

```bash
# From your local machine
scp <username>@ptolemy-login.arc.msstate.edu:/scratch/ptolemy/users/$USER/slurm-nanochat/report.md .
```

### 2. Analyze the Code

Study these key files:
- `nanochat/gpt.py` - Transformer implementation
- `scripts/base_train.py` - Training loop
- `nanochat/tokenizer.py` - BPE tokenization

### 3. Compare with Local CPU

Try: [Local CPU Quickstart](01-local-cpu-quickstart.md)

Then compare results!

### 4. Learn More

- [Training Environment Comparison](../explanation/training-environments.md)
- [HPC Environment Details](../explanation/hpc-environment.md)
- [Analyze Results](../how-to/analyze-results.md)

---

## File Locations

### On Ptolemy

```
/scratch/ptolemy/users/$USER/
├── slurm-nanochat/              # Repository
│   ├── logs/                    # SLURM job logs
│   ├── report.md                # Final report
│   └── .env.local               # Email config
│
├── nanochat-venv/               # Python virtual environment
│
└── nanochat-cache/              # All training artifacts
    ├── base_data/               # 240 shards (~24GB)
    ├── tokenizer/               # BPE tokenizer
    ├── models/                  # Model checkpoints
    │   ├── base_checkpoints/
    │   ├── mid_checkpoints/
    │   └── chatsft_checkpoints/
    ├── eval_bundle/             # Evaluation data
    └── report/                  # Training logs
```

---

## Quick Reference Commands

```bash
# Submit job
WANDB_RUN=my_run sbatch scripts/speedrun.slurm

# Check status
squeue -u $USER

# Watch logs
tail -f logs/nanochat_speedrun_*.out

# Interactive session
srun --account=class-cse8990 --partition=gpu-a100 --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash

# Chat
python -m scripts.chat_cli

# View report
cat report.md
```

---

**Congratulations!** You've successfully trained a production-scale ChatGPT model on an HPC cluster!

This is a complete, end-to-end LLM training pipeline - the same process used by AI labs, just at smaller scale.
