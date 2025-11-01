# How-To: Run a Training Job

This guide covers how to submit and monitor training jobs on all three platforms.

---

## Choose Your Platform

**Local CPU** - Run training on your laptop/desktop
→ [Jump to Local CPU Training](#local-cpu-training)

**Ptolemy HPC** - Submit SLURM job
→ [Jump to Ptolemy HPC Training](#ptolemy-hpc-training)

**Production GPU** - Run on cloud GPUs
→ [Jump to Production GPU Training](#production-gpu-training)

---

## Local CPU Training

### Quick Command

From the repository root:

```bash
bash scripts/local_cpu_train.sh
```

That's it! The script runs the complete pipeline automatically.

### What Happens

The script executes all training phases sequentially:

**Phase 1: Tokenizer Training** (~10-15 min)
```bash
# Downloads 4 data shards (~400MB)
# Trains BPE tokenizer on 1B characters
# Evaluates compression ratio
```

**Phase 2: Base Pretraining** (~30-60 min)
```bash
# Trains 4-layer model (~8M parameters)
# 50 optimization steps
# Evaluates CORE benchmark
```

**Phase 3: Midtraining** (~15-30 min)
```bash
# Teaches conversation format
# 100 optimization steps
# Evaluates chat benchmarks
```

**Phase 4: Supervised Finetuning** (~15-30 min)
```bash
# Improves instruction-following
# 100 optimization steps
# Re-evaluates all benchmarks
```

**Phase 5: Report Generation** (~1 min)
```bash
# Compiles all metrics
# Generates report.md
```

### Monitoring Progress

Watch the terminal output:

```bash
==================================================
[1/5] Training Tokenizer...
==================================================
Training tokenizer on ~1B characters...
Progress: [████████████████████] 100%

==================================================
[2/5] Base Model Pretraining...
==================================================
Training tiny d4 model (4 layers, ~8M parameters)...
step 1/50 | loss: 3.456 | time: 12.3s
step 2/50 | loss: 3.234 | time: 11.8s
...
```

### After Training Completes

**View Results:**
```bash
cat report.md
```

**Chat with Model:**
```bash
source .venv/bin/activate
python -m scripts.chat_cli
```

**Web Interface:**
```bash
source .venv/bin/activate
python -m scripts.chat_web
# Open http://localhost:8000
```

### Customizing Training

Edit `scripts/local_cpu_train.sh` to modify:

**Model Size:**
```bash
--depth=6              # More layers (default: 4)
--width=512            # Larger model dimension (default: 256)
```

**Training Duration:**
```bash
--num_iterations=100   # More steps (default: 50 for base)
```

**Data Amount:**
```bash
# In the download section, change:
python -m nanochat.dataset -n 8  # Download 8 shards instead of 4
```

**Context Length:**
```bash
--max_seq_len=2048     # Longer context (default: 1024)
```

### Stopping Training

If you need to stop:
- Press `Ctrl+C` to interrupt
- Training will stop at the current phase
- Partial checkpoints are saved

To resume, you'd need to modify the script to skip completed phases.

---

## Ptolemy HPC Training

### Critical Pre-Flight Check

Before submitting, ensure:
- ✅ On any Ptolemy login node (not devel, not compute)
- ✅ All data downloaded (ran `scripts/download_data.sh` on devel node)
- ✅ Environment set up (ran `scripts/setup_environment.sh`)
- ✅ Email configured (`.env.local` file exists)

### Submit Training Job

**The command:**

```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# CRITICAL: Set WANDB_RUN to any non-"dummy" name
WANDB_RUN=my_training_run sbatch scripts/speedrun.slurm
```

**Alternative with explicit email:**
```bash
WANDB_RUN=my_run sbatch --mail-user=your_email@msstate.edu scripts/speedrun.slurm
```

### WANDB_RUN Explained

**Why it matters:**
- Controls whether midtraining and SFT actually run
- If not set or set to "dummy", those phases are **SKIPPED**
- Chat functionality won't work without them

**Correct usage:**
```bash
# ✅ CORRECT - Any non-"dummy" name works
WANDB_RUN=ptolemy_run_1 sbatch scripts/speedrun.slurm
WANDB_RUN=my_first_training sbatch scripts/speedrun.slurm
WANDB_RUN=test_run sbatch scripts/speedrun.slurm
```

**Incorrect usage:**
```bash
# ❌ WRONG - Job will fail immediately
sbatch scripts/speedrun.slurm

# ❌ WRONG - Job will fail immediately
WANDB_RUN=dummy sbatch scripts/speedrun.slurm
```

**Note:** You do NOT need a wandb account! The system will skip wandb logging but still train everything.

### Job Submitted - What Happens Next

**Immediate:**
```
Submitted batch job 12345
```

**SLURM validates:**
1. WANDB_RUN is set correctly ✅
2. All required data exists ✅
3. Virtual environment exists ✅

**If validation fails:**
- Job terminates immediately
- Clear error message in log
- Fix issue and resubmit

**If validation passes:**
- Job enters queue (may wait)
- Requests 8xA100 GPUs
- Begins training when resources available

### Monitoring Your Job

**Check job status:**
```bash
squeue -u $USER
```

Output:
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
12345  gpu-a100 nanochat  yourid  PD       0:00      1 (Priority)
12345  gpu-a100 nanochat  yourid  R    1:23:45      1 ptolemy-gpu-01
```

Status codes:
- `PD` = Pending (waiting in queue)
- `R` = Running
- No output = Completed (check logs)

**Watch training live:**
```bash
# Get job ID from squeue
JOBID=12345

# Watch output
tail -f /scratch/ptolemy/users/$USER/slurm-nanochat/logs/nanochat_speedrun_${JOBID}.out
```

Press `Ctrl+C` to stop watching (job keeps running).

**Check for errors:**
```bash
tail -f /scratch/ptolemy/users/$USER/slurm-nanochat/logs/nanochat_speedrun_${JOBID}.err
```

### Training Timeline (~10-12 hours)

```
0:00  → Job starts
0:00  → Validation passes ✅
0:05  → Tokenizer evaluation complete
2:00  → Base pretraining ongoing... (20% done)
4:00  → Base pretraining ongoing... (40% done)
6:00  → Base pretraining ongoing... (60% done)
7:00  → Base pretraining complete ✅
7:30  → Midtraining ongoing...
9:30  → Midtraining complete ✅
10:00 → SFT ongoing...
11:30 → SFT complete ✅
11:35 → Report generated ✅
11:36 → Job done!
```

### Email Notifications

You'll receive emails at:
- **BEGIN** - Job started on compute node
- **END** - Job completed successfully
- **FAIL** - Job failed (check error log)

### After Job Completes

**View results:**
```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat
cat report.md
```

**Download to local machine:**
```bash
# From your local computer
scp <username>@ptolemy-login.arc.msstate.edu:/scratch/ptolemy/users/$USER/slurm-nanochat/report.md .
```

### Canceling a Job

If you need to stop training:

```bash
# Find job ID
squeue -u $USER

# Cancel it
scancel 12345
```

**Note:** Partial checkpoints may be saved, but you'll need to resubmit from the beginning.

---

## Production GPU Training

### Pre-Flight Check

- ✅ GPU instance running (8xH100)
- ✅ SSH access working
- ✅ Repository cloned

### Start Training

**Using screen (recommended):**

```bash
# Start screen session
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Detach with: Ctrl+a d
# Reattach with: screen -r speedrun
```

**Direct execution (not recommended for long jobs):**

```bash
bash speedrun.sh
```

This is risky - if SSH disconnects, training stops!

### What the Script Does

The `speedrun.sh` script:
1. Creates Python virtual environment
2. Installs all dependencies
3. Downloads 240 data shards (~24GB, ~30-60 min)
4. Trains tokenizer
5. Pretrains base model (~2-2.5 hours)
6. Runs midtraining (~30-45 min)
7. Runs SFT (~20-30 min)
8. Generates report

**Total time:** ~4 hours

### Monitoring Progress

**If using screen:**
```bash
# Reattach to session
screen -r speedrun

# Or view log file
tail -f speedrun.log
```

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check progress in log:**
```bash
grep "step.*loss" speedrun.log | tail -20
```

### Training Timeline (~4 hours)

```
0:00  → Setup and data download starts
0:45  → Data download complete
1:00  → Tokenizer training complete
1:15  → Base pretraining starts
3:30  → Base pretraining complete
4:00  → Midtraining complete
4:30  → SFT complete
4:35  → Report generated
```

### After Training Completes

**View results:**
```bash
cat report.md
```

**Chat via CLI:**
```bash
source .venv/bin/activate
python -m scripts.chat_cli
```

**Serve web interface:**
```bash
source .venv/bin/activate
python -m scripts.chat_web
```

Then access via public IP:
```
http://<instance-public-ip>:8000
```

### Stopping Training

**Graceful stop:**
```bash
# If using screen, reattach
screen -r speedrun

# Press Ctrl+C
```

**Force kill:**
```bash
pkill -f base_train.py
```

**Note:** You'll need to start over - no easy resume mechanism.

### Cost Management

**Monitor costs:**
- Training time: ~4 hours
- Cost: 8 GPUs × $3/hour × 4 hours = ~$96
- **Terminate instance immediately after training!**

**Download results before terminating:**
```bash
# From local machine
scp ubuntu@<instance-ip>:~/nanochat/report.md .
scp -r ubuntu@<instance-ip>:~/.cache/nanochat/models/ ./models/
```

---

## Comparison Table

| Aspect | Local CPU | Ptolemy HPC | Production GPU |
|--------|-----------|-------------|----------------|
| **Command** | `bash scripts/local_cpu_train.sh` | `WANDB_RUN=X sbatch scripts/speedrun.slurm` | `bash speedrun.sh` |
| **Monitoring** | Terminal output | `tail -f logs/*.out` | `tail -f speedrun.log` |
| **Duration** | 1-3 hours | ~12 hours | ~4 hours |
| **Can Disconnect?** | No (terminal) | Yes (SLURM job) | Yes (with screen) |
| **Cost** | Free | Free | ~$100 |
| **Queue Time** | None | Variable | None |
| **Stop/Resume** | Manual | Cancel/resubmit | Manual |

---

## Common Patterns

### Quick Test Run (Local CPU)

```bash
# Modify scripts/local_cpu_train.sh:
# - Reduce iterations to 10
# - Use only 1 data shard
# - Smaller model (depth=2)
bash scripts/local_cpu_train.sh
```

### Full Production Run (Ptolemy)

```bash
# Download all data on devel node first
ssh <username>@ptolemy-devel-1.arc.msstate.edu
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_data.sh

# Then submit job from any login node
ssh <username>@ptolemy-login.arc.msstate.edu
cd /scratch/ptolemy/users/$USER/slurm-nanochat
WANDB_RUN=production_run_1 sbatch scripts/speedrun.slurm
```

### Cloud GPU with Monitoring

```bash
# Start training in screen
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Detach: Ctrl+a d

# In another SSH session, monitor
tail -f speedrun.log
watch -n 5 nvidia-smi

# Check back periodically or when email arrives (if configured)
```

---

## Troubleshooting

### Job Fails Immediately

**Ptolemy:**
```bash
# Check error log
cat logs/nanochat_speedrun_*.err

# Common causes:
# - WANDB_RUN not set → Set it!
# - Data not downloaded → Run download_data.sh on devel node
# - No venv → Run setup_environment.sh
```

### Training Runs But Model is Bad

**Check report.md:**
- CORE score should be > 0.15 (local CPU) or > 0.20 (GPU)
- If much lower, training may have failed silently
- Check logs for errors

### Out of Memory (OOM)

**Reduce batch size:**
```bash
# Edit the training script
--device_batch_size=16  # Instead of 32
```

### Training Too Slow

**Local CPU:**
- Expected! CPU is ~30x slower than GPU
- Reduce model size or iterations

**Ptolemy/Production:**
- Check GPU utilization: `nvidia-smi`
- Should be near 100% during training

---

## Next Steps

After training completes:

→ [Analyze Results](analyze-results.md)
→ [Troubleshoot Issues](troubleshoot-common-issues.md)

For environment setup:
→ [Setup Environment](setup-environment.md)
