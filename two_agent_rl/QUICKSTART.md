# Quick Start Guide - Two-Agent RL Decoder

This guide will get you running experiments in 5 minutes!

## 📦 Installation (1 minute)

```bash
cd two_agent_rl

# Install dependencies
pip install torch torch-geometric numpy scipy matplotlib

# Install LDPC library for BP-OSD baseline
pip install ldpc
```

## 🚀 Three Ways to Run Experiments

### Option 1: Quick Demo (5 minutes) ⚡

Test everything with minimal training (good for checking if it works):

```bash
python run_experiments.py --mode quick
```

This will:
- ✅ Train two-agent decoder for 5,000 steps (~2 min)
- ✅ Evaluate on 2 error rates with 50 trials each (~2 min)
- ✅ Compare with BP-OSD baseline
- ✅ Generate plots and reports
- ✅ Total time: ~5 minutes

**Output:**
```
./results/experiment_YYYYMMDD_HHMMSS/
├── comparison_results.json      # Numerical results
├── comparison_report.txt        # Text summary
├── error_rate_comparison.png    # LER plot
├── success_rate_comparison.png  # Success rate plot
└── two_agent_final.pt          # Trained model
```

### Option 2: Full Experiment (2-4 hours) 🔬

Complete training and comprehensive evaluation:

```bash
python run_experiments.py --mode full
```

This will:
- ✅ Train two-agent decoder for 100,000 steps (~1-2 hours)
- ✅ Evaluate on 5 error rates with 1,000 trials each (~1-2 hours)
- ✅ Compare with BP-OSD and hybrid decoder
- ✅ Generate comprehensive plots

**Recommended for:** Publication-quality results

### Option 3: Evaluate Only (10 minutes) 📊

If you already have trained models:

```bash
python run_experiments.py \
    --mode evaluate \
    --load_two_agent ./results/experiment_XXX/two_agent_final.pt
```

This will:
- ✅ Load pretrained model
- ✅ Evaluate on multiple error rates
- ✅ Generate comparison plots
- ✅ No training required!

## 📊 Understanding the Results

After running, you'll see output like this:

```
=== DECODER COMPARISON SUMMARY ===

Error Rate p = 0.0010
--------------------------------------------------------------------------------
Decoder              LER          Success      Time (ms)    Syn Sat
--------------------------------------------------------------------------------
bposd                0.0023       0.9977       0.52         0.9977
two_agent            0.0015       0.9985       1.48         0.9985
hybrid               0.0014       0.9986       0.68         0.9986
```

**What this means:**
- **LER**: Logical Error Rate (lower is better) ← **Most important metric**
- **Success**: Successful decoding rate (higher is better)
- **Time**: Average decode time in milliseconds
- **Syn Sat**: Syndrome satisfaction rate

**Key insight:** Two-agent RL achieves lower LER than BP-OSD but is slower. Hybrid combines both advantages!

## 🎨 Visualizations Generated

### 1. Error Rate Comparison (`error_rate_comparison.png`)

Shows logical error rate vs physical error rate for each decoder.

**What to look for:**
- Two-agent should be below BP-OSD (better accuracy)
- Hybrid should be close to two-agent (best performance)
- Gap increases at higher error rates

### 2. Success Rate Comparison (`success_rate_comparison.png`)

Shows successful decoding percentage vs error rate.

**What to look for:**
- All decoders should have high success rate (>95%) at low error rates
- Two-agent should maintain high success at higher error rates

## ⚙️ Advanced Usage

### Custom Training Parameters

```bash
python run_experiments.py \
    --mode full \
    --two_agent_timesteps 200000 \
    --hidden_dim 256 \
    --num_gnn_layers 6 \
    --lr 1e-4
```

### Test Different Code Sizes

```bash
# Smaller code (faster training)
python run_experiments.py --mode quick --m 3 --ell 6

# Larger code (more qubits)
python run_experiments.py --mode full --m 12 --ell 12
```

### GPU Acceleration

```bash
python run_experiments.py --mode full --device cuda
```

### Only Compare with BP-OSD (Skip Single-Agent)

```bash
python run_experiments.py \
    --mode full \
    --include_single_agent False
```

## 🔍 Step-by-Step Manual Execution

If you want more control, run each step manually:

### Step 1: Train Two-Agent Decoder

```bash
python train.py \
    --decoder_type two_agent \
    --total_timesteps 50000 \
    --error_rate 0.001 \
    --save_path ./checkpoints/my_experiment
```

**Monitor training:**
```
--- Update 50 ---
Timesteps: 2500/50000
Mean reward: 42.15
Success rate: 78.3%
Logical error rate: 18.7%
Policy loss (L/R): 0.0234 / 0.0219
```

### Step 2: Evaluate Manually

```python
from experiments import DecoderBenchmark
from agent_architecture import TwoAgentDecoder
import torch

# Load trained model
decoder = TwoAgentDecoder(m=6, ell=12)
checkpoint = torch.load('./checkpoints/my_experiment/final_model.pt')
decoder.agent_left.load_state_dict(checkpoint['agent_left'])
decoder.agent_right.load_state_dict(checkpoint['agent_right'])

# Run benchmark
benchmark = DecoderBenchmark(m=6, ell=12)
results = benchmark.evaluate_decoder(
    decoder,
    decoder_type='two_agent',
    error_rates=[0.0001, 0.001, 0.01],
    num_trials=1000
)

# Print results
for i, p in enumerate(results['error_rates']):
    print(f"p={p:.4f}: LER={results['logical_error_rates'][i]:.4f}")
```

### Step 3: Compare Multiple Decoders

```python
from experiments import DecoderBenchmark

benchmark = DecoderBenchmark(m=6, ell=12)

comparison = benchmark.compare_decoders(
    two_agent_decoder=decoder,
    error_rates=[0.0001, 0.0005, 0.001, 0.002, 0.005],
    num_trials=1000,
    include_bposd=True
)

benchmark.print_summary(comparison)
benchmark.save_results(comparison, 'my_comparison.json')
```

### Step 4: Visualize

```python
from visualize import DecoderComparison, load_results

results = load_results('my_comparison.json')
vis = DecoderComparison()

vis.plot_error_rate_comparison(results, save_path='my_plot.png')
```

## 🐛 Troubleshooting

### Error: "No module named 'ldpc'"

```bash
pip install ldpc
```

If that fails:
```bash
# Install dependencies for ldpc
pip install cython
pip install git+https://github.com/quantumgizmos/ldpc.git
```

### Error: "CUDA out of memory"

Use CPU instead:
```bash
python run_experiments.py --mode quick --device cpu
```

Or reduce batch size:
```bash
python train.py --episodes_per_update 5  # Default is 10
```

### Training is very slow

Reduce number of timesteps:
```bash
python run_experiments.py --mode quick --two_agent_timesteps 1000
```

Or use smaller network:
```bash
python train.py --hidden_dim 64 --num_gnn_layers 2
```

### Results look random / No improvement

This is normal for very short training! Solutions:

1. **Train longer**: Use `--mode full` instead of `quick`
2. **Check convergence**: Look at training curves - reward should increase
3. **Adjust hyperparameters**: Try lower learning rate `--lr 1e-4`

## 📈 Expected Performance

After proper training (100k+ timesteps), you should see:

| Error Rate | BP-OSD LER | Two-Agent LER | Improvement |
|------------|------------|---------------|-------------|
| 0.0001     | ~0.0001    | ~0.00008      | ~20% better |
| 0.001      | ~0.002     | ~0.0015       | ~25% better |
| 0.005      | ~0.015     | ~0.010        | ~33% better |

**Note:** Exact numbers depend on training time and hyperparameters.

## 🎯 What to Do Next

### For Research:
1. Run full experiments: `python run_experiments.py --mode full`
2. Test different code sizes
3. Analyze coordination patterns
4. Compare with other decoders

### For Production:
1. Train with diverse error rates (curriculum learning)
2. Use hybrid decoder for best speed/accuracy tradeoff
3. Benchmark on realistic noise models
4. Optimize inference speed

### For Understanding:
1. Run quick demo: `python run_experiments.py --mode quick`
2. Read the theoretical analysis: `THEORETICAL_ANALYSIS.md`
3. Check example usage: `python example_usage.py`
4. Visualize decoding process (see `visualize.py`)

## 📝 Summary of Commands

```bash
# Quick test (5 min)
python run_experiments.py --mode quick

# Full experiment (2-4 hours)
python run_experiments.py --mode full

# Evaluate pretrained model (10 min)
python run_experiments.py --mode evaluate --load_two_agent ./model.pt

# Custom training
python train.py --total_timesteps 100000 --error_rate 0.001

# On GPU
python run_experiments.py --mode full --device cuda

# Smaller code (faster)
python run_experiments.py --mode quick --m 3 --ell 6
```

## 🆘 Need Help?

1. Check `README.md` for detailed documentation
2. See `example_usage.py` for code examples
3. Read `THEORETICAL_ANALYSIS.md` for theory
4. Check `SUMMARY.md` for implementation overview

Happy experimenting! 🚀
