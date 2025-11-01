# Local CPU Training Documentation Index

This folder contains all documentation specific to running nanochat **locally on CPU** for development, testing, and analysis without requiring GPU access.

---

## 🚀 Quick Start

### New to Local CPU Training?
1. **[START_HERE_LOCAL_CPU.md](START_HERE_LOCAL_CPU.md)** ⭐ **START HERE** - Complete beginner's guide

### Want Quick Setup?
1. **[LOCAL_CPU_QUICKSTART.md](LOCAL_CPU_QUICKSTART.md)** - Fast setup for experienced users

---

## 📚 Documentation by Purpose

### Setup & Getting Started
- **[START_HERE_LOCAL_CPU.md](START_HERE_LOCAL_CPU.md)** - Beginner-friendly complete guide
- **[LOCAL_CPU_QUICKSTART.md](LOCAL_CPU_QUICKSTART.md)** - Quick setup for experienced users
- **[LOCAL_CPU_README.md](LOCAL_CPU_README.md)** - Overview and introduction

### Training Guides
- **[LOCAL_CPU_TRAINING.md](LOCAL_CPU_TRAINING.md)** - Detailed training guide with configurations
- **[LOCAL_CPU_ANALYSIS_GUIDE.md](LOCAL_CPU_ANALYSIS_GUIDE.md)** - Analysis and experimentation guide

### Reference & Comparison
- **[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)** - GPU vs CPU comparison
- **[TROUBLESHOOTING_LOCAL_CPU.md](TROUBLESHOOTING_LOCAL_CPU.md)** - Common issues and solutions

### Status
- **[SESSION_SUMMARY_LOCAL_CPU.md](SESSION_SUMMARY_LOCAL_CPU.md)** - Session work summary

---

## 🎯 Common Tasks

### Train Tiny Model (Fastest)
```bash
python -m scripts.local_cpu_train --depth=4 --width=256 --vocab_size=512 --device_batch_size=4
```

Time: ~5 minutes
See: [LOCAL_CPU_QUICKSTART.md](LOCAL_CPU_QUICKSTART.md)

### Train Small Model (Recommended)
```bash
python -m scripts.local_cpu_train --depth=6 --width=384 --vocab_size=2048 --device_batch_size=2
```

Time: ~30 minutes
See: [LOCAL_CPU_TRAINING.md](LOCAL_CPU_TRAINING.md)

### Analyze Results
```bash
python -m scripts.local_cpu_analysis
```

See: [LOCAL_CPU_ANALYSIS_GUIDE.md](LOCAL_CPU_ANALYSIS_GUIDE.md)

---

## 💡 Why Local CPU Training?

### Advantages
✅ **No GPU required** - Run on any laptop/desktop
✅ **Fast iteration** - Test changes in minutes
✅ **No queue time** - Start immediately
✅ **Free** - No compute costs
✅ **Easy debugging** - Full control and visibility

### Use Cases
- 🔬 **Experimentation** - Test hyperparameters
- 📚 **Learning** - Understand the code
- 🐛 **Debugging** - Fix issues before GPU run
- 📊 **Analysis** - Generate visualizations
- 🎓 **Assignment work** - Without cluster access

See: [LOCAL_CPU_README.md](LOCAL_CPU_README.md)

---

## ⚙️ Model Size Tiers

### Tiny (5 min)
```bash
--depth=4 --width=256 --vocab_size=512
```
- Parameters: ~1M
- Use: Quick testing

### Small (30 min)
```bash
--depth=6 --width=384 --vocab_size=2048
```
- Parameters: ~5M
- Use: Experimentation

### Medium (2-3 hours)
```bash
--depth=8 --width=512 --vocab_size=4096
```
- Parameters: ~15M
- Use: Serious local training

### Large (8+ hours)
```bash
--depth=12 --width=768 --vocab_size=8192
```
- Parameters: ~50M
- Use: Maximum CPU capability

See: [TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)

---

## 🔧 Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Reduce batch size
--device_batch_size=1

# Or reduce model size
--width=256 --vocab_size=512
```

**Training Too Slow:**
```bash
# Use tiny model
--depth=4 --width=256

# Reduce iterations
--num_iterations=100
```

**Can't Find Script:**
```bash
# Ensure in project root
cd /path/to/slurm-nanochat

# Activate venv
source .venv/bin/activate
```

See: [TROUBLESHOOTING_LOCAL_CPU.md](TROUBLESHOOTING_LOCAL_CPU.md)

---

## 📊 Performance Expectations

### Training Speed (CPU vs GPU)
| Model Size | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| Tiny (1M)  | 5 min    | 10 sec   | 30x     |
| Small (5M) | 30 min   | 1 min    | 30x     |
| Medium (15M)| 3 hours | 6 min    | 30x     |
| Large (50M)| 12 hours | 24 min   | 30x     |
| Full (560M)| N/A      | 7 hours  | N/A     |

See: [TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)

---

## 🎓 Learning Path

### Beginner
1. Read [START_HERE_LOCAL_CPU.md](START_HERE_LOCAL_CPU.md)
2. Train tiny model (5 min)
3. Experiment with parameters
4. Read analysis guide

### Intermediate
1. Read [LOCAL_CPU_TRAINING.md](LOCAL_CPU_TRAINING.md)
2. Train small model (30 min)
3. Analyze results
4. Compare with GPU version

### Advanced
1. Train medium model (2-3 hours)
2. Custom configurations
3. Read [LOCAL_CPU_ANALYSIS_GUIDE.md](LOCAL_CPU_ANALYSIS_GUIDE.md)
4. Generate visualizations

---

## 📁 File Organization

### This Folder (local_cpu_docs/)
All local CPU training documentation

### Parent Directory
- `scripts/local_cpu_train.py` - Training script
- `scripts/local_cpu_analysis.py` - Analysis script
- `README.md` - Main project README

### Other Docs
- `ptolemy_slurm_docs/` - Ptolemy/SLURM documentation

---

## 🔬 Experimentation Ideas

### Hyperparameter Sweeps
```bash
# Try different depths
for depth in 4 6 8; do
    python -m scripts.local_cpu_train --depth=$depth
done

# Try different widths
for width in 256 384 512; do
    python -m scripts.local_cpu_train --width=$width
done
```

### Learning Rate Experiments
```bash
# Different learning rates
python -m scripts.local_cpu_train --matrix_lr=0.01
python -m scripts.local_cpu_train --matrix_lr=0.02
python -m scripts.local_cpu_train --matrix_lr=0.04
```

See: [LOCAL_CPU_ANALYSIS_GUIDE.md](LOCAL_CPU_ANALYSIS_GUIDE.md)

---

## 💻 System Requirements

### Minimum
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB
- Python: 3.10+

### Recommended
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 20GB
- Python: 3.12

### For Large Models
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB

See: [LOCAL_CPU_README.md](LOCAL_CPU_README.md)

---

## 🎯 Assignment Use

### For MSU CSE8990 Students
If you can't access Ptolemy or want to experiment locally:

1. **Quick Test:** Train tiny model in 5 minutes
2. **Analysis:** Generate plots and metrics
3. **Code Study:** Understand architecture
4. **Comparison:** Compare with GPU results

**Note:** Local CPU training is great for learning, but GPU training on Ptolemy is recommended for the full experience.

---

**Last Updated:** 2025-11-01
**Status:** Fully functional, ready for use ✅
