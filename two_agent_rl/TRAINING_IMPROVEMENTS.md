# Training Improvements Summary

## Changes Made

### Increased Training Timesteps to 100,000

The training duration has been increased from 5,000 to 100,000 timesteps to allow the RL agents to learn more effectively.

### Files Modified:

1. **`run_experiments.py`** (lines 308-309)
   - Changed `quick` mode default from 5,000 to 100,000 timesteps
   - Both two-agent and single-agent now train for 100,000 timesteps in quick mode
   
2. **`example_usage.py`** (line 63)
   - Changed example training from 10,000 to 100,000 timesteps
   - Updated comment to reflect the change

## How to Run Improved Training

### Option 1: Use Quick Mode (now with 100k timesteps)
```bash
cd /Users/zhouy37/BivariateBicycleCodes/two_agent_rl
python run_experiments.py --mode quick
```

This will:
- Train two-agent decoder for 100,000 timesteps
- Train single-agent decoder for 100,000 timesteps
- Evaluate both against BP-OSD and hybrid decoder
- Generate comparison plots

### Option 2: Use Full Mode (100k timesteps + more evaluation)
```bash
python run_experiments.py --mode full
```

This will:
- Train for 100,000 timesteps (same as quick)
- Run 1,000 evaluation trials per error rate (vs 50 in quick)
- Test on more error rates: [0.0001, 0.0005, 0.001, 0.002, 0.005]

### Option 3: Custom Timesteps
```bash
python run_experiments.py --two_agent_timesteps 200000 --single_agent_timesteps 200000
```

## Expected Improvements

With 100,000 timesteps (20x increase), you should see:

### Two-Agent Decoder:
- **Correction weight**: Should decrease from ~130 to ~7-13 qubits
- **Syndrome satisfaction**: Should increase from 8% to 50%+ 
- **Logical error rate**: Should decrease from 100% to <50%
- **Success rate**: Should increase from 0% to 20%+

### Single-Agent Decoder:
- **Correction weight**: Should increase from ~1-1.6 to ~7-13 qubits
- **Logical error rate**: Should decrease from 86% to <40%
- **Success rate**: Should increase from 14% to 30%+

## Further Improvements

For even better performance, consider:

1. **More training timesteps**: 500,000 - 1,000,000 timesteps
   ```bash
   python run_experiments.py --two_agent_timesteps 500000
   ```

2. **Curriculum learning**: Train on progressively harder error rates
   - Modify the training script to start with p=0.0001 and gradually increase

3. **Improved reward function**: 
   - Check `environment.py` line ~350-400 for reward calculation
   - Increase penalties for syndrome violations
   - Add stronger rewards for successful decoding

4. **Entropy bonus**: Increase exploration
   ```bash
   # Would require modifying training.py to expose entropy_coef parameter
   ```

5. **Supervised pretraining**: 
   - Pretrain on BP-OSD solutions before RL training
   - This gives agents a good starting point

## Monitoring Training Progress

Training logs will show:
- Episode rewards (should increase over time)
- Syndrome satisfaction rate (should increase)
- Correction weights (should stabilize around 7-13)

Look for checkpoints in:
```
./results/experiment_YYYYMMDD_HHMMSS/two_agent/checkpoint_*.pt
```

Best model saved as:
```
./results/experiment_YYYYMMDD_HHMMSS/two_agent_final.pt
```

## Estimated Training Time

- **Quick mode** (100k timesteps): ~2-4 hours on CPU
- **Full mode** (100k timesteps): ~3-5 hours on CPU (more evaluation)
- **500k timesteps**: ~10-20 hours on CPU

Using GPU would speed this up significantly if available.

## Next Steps

1. Run the improved training:
   ```bash
   python run_experiments.py --mode quick
   ```

2. Monitor the output for:
   - Increasing episode rewards
   - Decreasing logical error rates
   - Improving syndrome satisfaction

3. If results are still poor after 100k timesteps:
   - Increase to 500k timesteps
   - Check the reward function in `environment.py`
   - Consider supervised pretraining

4. Compare results:
   - New plots will be generated in `./results/experiment_*/`
   - Compare to previous results in `experiment_20251029_015506/`

