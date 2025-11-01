# 🚀 START HERE: Local CPU Training

## Welcome!

You want to understand the nanochat training pipeline by running it locally on your CPU. You're in the right place!

## ⚡ Ultra-Quick Start

**One command does everything:**

```bash
bash scripts/local_cpu_train.sh
```

**That's it!** Wait 1-3 hours and you'll have trained a complete chatbot.

## 📖 What to Read

### If you want to jump right in:
👉 **[LOCAL_CPU_QUICKSTART.md](LOCAL_CPU_QUICKSTART.md)** - 5 minute read, get started immediately

### If you want all the details:
👉 **[LOCAL_CPU_TRAINING.md](LOCAL_CPU_TRAINING.md)** - Complete guide with explanations

### After training, to understand the code:
👉 **[LOCAL_CPU_ANALYSIS_GUIDE.md](LOCAL_CPU_ANALYSIS_GUIDE.md)** - Step-by-step code walkthrough

### To compare CPU vs GPU vs HPC:
👉 **[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)** - See all the differences

### For a complete overview:
👉 **[LOCAL_CPU_README.md](LOCAL_CPU_README.md)** - Index of everything

## 🎯 What You'll Get

After running `scripts/local_cpu_train.sh`:

✅ A complete trained model (all phases: tokenizer → base → midtraining → SFT)
✅ A training report with all metrics
✅ A working chatbot you can talk to
✅ Deep understanding of the training pipeline
✅ Experience with all training phases
✅ Knowledge to scale up to production

## ⚠️ Important Notes

**Your model will NOT be very smart!** This is expected and intentional:
- You're training a tiny 4-layer model (~8M parameters)
- On minimal data (400MB vs 24GB for production)
- For only 50 pretraining steps (vs ~5000 for production)
- **The goal is LEARNING, not production**

## 📋 What Happens During Training

```
Phase 1: Tokenizer Training      (~10-15 min)
   ↓
Phase 2: Base Pretraining         (~30-60 min)
   ↓
Phase 3: Midtraining              (~15-30 min)
   ↓
Phase 4: Supervised Finetuning    (~15-30 min)
   ↓
Phase 5: Report Generation        (~1 min)
   ↓
✅ DONE! You have a trained chatbot
```

## 🎮 After Training

### Chat with your model:
```bash
source .venv/bin/activate
python -m scripts.chat_cli -p "Tell me a joke"
```

### Or use web interface:
```bash
source .venv/bin/activate
python -m scripts.chat_web
# Open: http://localhost:8000
```

### Read the training report:
```bash
cat report.md
```

## 🔍 Key Differences from Production

| | Local CPU | Production GPU |
|---|---|---|
| **Time** | 1-3 hours | 4 hours |
| **Cost** | Free | ~$100 |
| **Quality** | Poor (learning only) | Good (outperforms GPT-2) |
| **Model** | 4 layers, 8M params | 20 layers, 561M params |
| **Data** | 400MB | 24GB |

## 📁 Files Created for You

### The Training Script:
- **`scripts/local_cpu_train.sh`** - Run this to train locally

### Documentation:
- **`START_HERE_LOCAL_CPU.md`** - This file (you are here!)
- **`LOCAL_CPU_QUICKSTART.md`** - Quick reference
- **`LOCAL_CPU_TRAINING.md`** - Complete guide
- **`LOCAL_CPU_ANALYSIS_GUIDE.md`** - Code analysis
- **`TRAINING_COMPARISON.md`** - CPU vs GPU vs HPC
- **`LOCAL_CPU_README.md`** - Overview and index

### Unchanged (your GPU/HPC workflows still work):
- ✅ `speedrun.sh` - Production GPU training (unchanged)
- ✅ `scripts/speedrun.slurm` - Ptolemy HPC training (unchanged)
- ✅ `PTOLEMY_SETUP.md` - HPC documentation (unchanged)

## 🚀 Next Steps

### 1️⃣ Run the training:
```bash
bash scripts/local_cpu_train.sh
```

### 2️⃣ While it runs, read:
- **LOCAL_CPU_QUICKSTART.md** - Understand what's happening
- **LOCAL_CPU_TRAINING.md** - Learn about each phase

### 3️⃣ After training completes:
- Read `report.md` - Your training metrics
- Chat with your model - See it in action
- Follow **LOCAL_CPU_ANALYSIS_GUIDE.md** - Understand the code

### 4️⃣ Compare and scale:
- Read **TRAINING_COMPARISON.md** - See all options
- Scale to GPU when ready - `bash speedrun.sh` or Ptolemy HPC

## ❓ Questions?

### "Will this work on my machine?"
✅ Yes! Works on any CPU (Linux, macOS, Windows/WSL)
- Needs: Python 3.10+, 8GB+ RAM, 2GB disk space

### "How long will it take?"
⏱️ 1-3 hours depending on your CPU speed

### "Will my model be good?"
❌ No, it will be tiny and make lots of mistakes
✅ But you'll understand the ENTIRE training pipeline!

### "Can I modify the code?"
✅ Yes! All scripts are separate from production
✅ Experiment freely with `scripts/local_cpu_train.sh`

### "What if something breaks?"
📚 Check the troubleshooting sections in:
- LOCAL_CPU_TRAINING.md (detailed troubleshooting)
- LOCAL_CPU_QUICKSTART.md (quick fixes)

**Common issues:**
- `uv: command not found` → Script auto-fixes, or run `export PATH="$HOME/.local/bin:$PATH"`
- Rust `edition2024` error → Run `rustup update` to get Rust 1.93.0+ nightly

### "Should I use CPU or GPU training?"
- **Use CPU** for learning and understanding
- **Use GPU** when you need a good model
- See **TRAINING_COMPARISON.md** for details

## 🎯 Success Looks Like

You'll know you succeeded when:

✅ Training completes without errors
✅ You can chat with your model
✅ You understand each training phase
✅ You can explain what the report metrics mean
✅ You're ready to read and modify the code
✅ You can scale to GPU training if needed

## 🎓 Learning Path

```
1. Run: bash scripts/local_cpu_train.sh
        ↓
2. Read: LOCAL_CPU_QUICKSTART.md (while training)
        ↓
3. Review: cat report.md (after training)
        ↓
4. Chat: python -m scripts.chat_cli (test your model)
        ↓
5. Analyze: Follow LOCAL_CPU_ANALYSIS_GUIDE.md
        ↓
6. Compare: Read TRAINING_COMPARISON.md
        ↓
7. Scale: Try GPU training or keep experimenting
```

## 💡 Pro Tips

1. **Run in screen/tmux** so you can detach:
   ```bash
   screen -S nanochat bash scripts/local_cpu_train.sh
   # Detach: Ctrl+a d
   # Reattach: screen -r nanochat
   ```

2. **Read docs while training** - Makes the 1-3 hours productive

3. **Check report.md first** - Gives you context for code analysis

4. **Try the web interface** - More fun than CLI

5. **Experiment after success** - Modify parameters, see what changes

## 🎉 Ready to Start?

### Recommended first command:
```bash
bash scripts/local_cpu_train.sh
```

### Recommended first reading:
**[LOCAL_CPU_QUICKSTART.md](LOCAL_CPU_QUICKSTART.md)**

---

## 📚 Quick Reference

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **START_HERE_LOCAL_CPU.md** | This file - overview | Right now! |
| **LOCAL_CPU_QUICKSTART.md** | Quick start guide | Before/during training |
| **LOCAL_CPU_TRAINING.md** | Complete details | During training |
| **LOCAL_CPU_ANALYSIS_GUIDE.md** | Code walkthrough | After training |
| **TRAINING_COMPARISON.md** | CPU vs GPU vs HPC | When planning next steps |
| **LOCAL_CPU_README.md** | Index and overview | Reference anytime |

---

## 🚀 Let's Go!

Run this command and start learning:

```bash
bash scripts/local_cpu_train.sh
```

**Happy learning!** 🎓

---

**Questions?** Read the guides linked above.

**Ready for more?** Check out the main `README.md` for production training.

**Need HPC?** See `PTOLEMY_SETUP.md` for Ptolemy cluster setup.
