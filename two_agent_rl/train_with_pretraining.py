#!/usr/bin/env python3
"""
Training Script with Supervised Pretraining

Compares three training strategies:
1. From scratch (baseline)
2. Supervised pretraining only
3. Mixed (supervised + RL fine-tuning)

Usage:
    # Train with pretraining (recommended)
    python train_with_pretraining.py --strategy mixed

    # Train from scratch (baseline)
    python train_with_pretraining.py --strategy scratch

    # Only supervised (fast, but limited)
    python train_with_pretraining.py --strategy supervised_only

    # Compare all three
    python train_with_pretraining.py --strategy compare_all
"""

import argparse
import torch
import os
import json
import time
from datetime import datetime

from decoder_setup import bivariate_bicycle_codes
from environment import BBCodeDecodingEnv
from agent_architecture import TwoAgentDecoder
from training import MultiAgentPPO
from supervised_pretraining import BPOSDExpertDataset, SupervisedPretrainer, MixedTrainer
from bposd_panel_features import analyze_bposd_panel_statistics
from experiments import DecoderBenchmark
from visualize import DecoderComparison


def train_from_scratch(args, save_path):
    """Train two-agent decoder from scratch (no pretraining)."""
    print("\n" + "=" * 70)
    print("STRATEGY 1: TRAINING FROM SCRATCH")
    print("=" * 70)

    # Create environment
    env = BBCodeDecodingEnv(
        m=args.m,
        ell=args.ell,
        num_cycles=args.num_cycles,
        error_rate=args.error_rate,
        reward_type='mixed',
        max_steps=5
    )

    # Create decoder
    decoder = TwoAgentDecoder(
        m=args.m,
        ell=args.ell,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers
    )

    # Create trainer
    trainer = MultiAgentPPO(
        env=env,
        decoder=decoder,
        lr=args.rl_lr,
        device=args.device
    )

    # Train
    start_time = time.time()
    trainer.train(
        total_timesteps=args.rl_timesteps,
        episodes_per_update=10,
        log_interval=10,
        save_interval=50,
        save_path=os.path.join(save_path, 'from_scratch')
    )
    train_time = time.time() - start_time

    # Save model
    model_path = os.path.join(save_path, 'from_scratch_final.pt')
    torch.save({
        'agent_left': decoder.agent_left.state_dict(),
        'agent_right': decoder.agent_right.state_dict(),
        'train_time': train_time,
        'final_success_rate': list(trainer.success_rate)[-1] if trainer.success_rate else 0
    }, model_path)

    print(f"\nModel trained from scratch in {train_time:.1f}s")
    return decoder, model_path


def train_supervised_only(args, code_data, env, save_path):
    """Train with supervised learning only (no RL)."""
    print("\n" + "=" * 70)
    print("STRATEGY 2: SUPERVISED LEARNING ONLY")
    print("=" * 70)

    # Create decoder
    decoder = TwoAgentDecoder(
        m=args.m,
        ell=args.ell,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers
    )

    # Generate dataset
    dataset = BPOSDExpertDataset(
        code_data,
        error_rate=args.error_rate,
        num_samples=args.supervised_samples
    )
    dataset.generate_dataset()

    # Pretrain
    pretrainer = SupervisedPretrainer(
        decoder, dataset, env,
        lr=args.supervised_lr,
        device=args.device
    )

    start_time = time.time()
    pretrainer.pretrain(
        num_epochs=args.supervised_epochs,
        batch_size=32,
        batches_per_epoch=100
    )
    train_time = time.time() - start_time

    # Save model
    model_path = os.path.join(save_path, 'supervised_only_final.pt')
    torch.save({
        'agent_left': decoder.agent_left.state_dict(),
        'agent_right': decoder.agent_right.state_dict(),
        'train_time': train_time,
        'final_accuracy': pretrainer.train_accuracies[-1]
    }, model_path)

    print(f"\nSupervised model trained in {train_time:.1f}s")
    return decoder, model_path


def train_mixed(args, code_data, env, save_path):
    """Train with mixed strategy (supervised + RL)."""
    print("\n" + "=" * 70)
    print("STRATEGY 3: MIXED (SUPERVISED PRETRAINING + RL FINE-TUNING)")
    print("=" * 70)

    # Create decoder
    decoder = TwoAgentDecoder(
        m=args.m,
        ell=args.ell,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers
    )

    # Mixed trainer
    mixed_trainer = MixedTrainer(
        code_data, decoder, env, device=args.device
    )

    start_time = time.time()
    mixed_trainer.train(
        supervised_samples=args.supervised_samples,
        supervised_epochs=args.supervised_epochs,
        supervised_lr=args.supervised_lr,
        rl_timesteps=args.rl_timesteps,
        rl_lr=args.rl_lr,
        save_dir=os.path.join(save_path, 'mixed')
    )
    train_time = time.time() - start_time

    model_path = os.path.join(save_path, 'mixed', 'final_mixed.pt')
    print(f"\nMixed model trained in {train_time:.1f}s")
    return decoder, model_path


def evaluate_and_compare(decoders_dict, args, save_path):
    """Evaluate and compare all trained models."""
    print("\n" + "=" * 70)
    print("EVALUATION AND COMPARISON")
    print("=" * 70)

    benchmark = DecoderBenchmark(m=args.m, ell=args.ell, device=args.device)

    error_rates = [args.error_rate * 0.5, args.error_rate, args.error_rate * 2]
    num_trials = 200 if args.quick else 500

    comparison = {}

    for name, decoder in decoders_dict.items():
        if decoder is None:
            continue

        print(f"\nEvaluating: {name}")
        results = benchmark.evaluate_decoder(
            decoder,
            'two_agent',
            error_rates,
            num_trials,
            verbose=True
        )
        comparison[name] = results

    # Print comparison
    benchmark.print_summary(comparison)

    # Save results
    results_path = os.path.join(save_path, 'pretraining_comparison.json')
    benchmark.save_results(comparison, results_path)

    # Visualize
    vis = DecoderComparison()
    vis.plot_error_rate_comparison(
        comparison,
        save_path=os.path.join(save_path, 'pretraining_comparison.png')
    )

    return comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train two-agent decoder with optional supervised pretraining'
    )

    parser.add_argument('--strategy', type=str, default='mixed',
                        choices=['scratch', 'supervised_only', 'mixed', 'compare_all'],
                        help='Training strategy')

    # Code parameters
    parser.add_argument('--m', type=int, default=6)
    parser.add_argument('--ell', type=int, default=12)
    parser.add_argument('--num_cycles', type=int, default=3)
    parser.add_argument('--error_rate', type=float, default=0.001)

    # Network
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_gnn_layers', type=int, default=4)

    # Supervised pretraining
    parser.add_argument('--supervised_samples', type=int, default=5000,
                        help='Number of BP-OSD demonstrations')
    parser.add_argument('--supervised_epochs', type=int, default=10)
    parser.add_argument('--supervised_lr', type=float, default=1e-3)

    # RL training
    parser.add_argument('--rl_timesteps', type=int, default=25000)
    parser.add_argument('--rl_lr', type=float, default=3e-4)

    # Other
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_path', type=str, default='./results/pretraining')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer samples for testing)')
    parser.add_argument('--analyze_bposd', action='store_true',
                        help='Analyze BP-OSD panel statistics first')

    args = parser.parse_args()

    if args.quick:
        args.supervised_samples = 1000
        args.supervised_epochs = 5
        args.rl_timesteps = 5000

    return args


def main():
    args = parse_args()

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(args.save_path, f'experiment_{timestamp}')
    os.makedirs(save_path, exist_ok=True)

    # Save config
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 70)
    print("SUPERVISED PRETRAINING EXPERIMENT")
    print("=" * 70)
    print(f"Strategy: {args.strategy}")
    print(f"Code: [[{2*args.m*args.ell}, ?, ?]]")
    print(f"Save path: {save_path}")
    print("=" * 70)

    # Initialize code and environment
    code_data = bivariate_bicycle_codes(
        args.m, args.ell, (3, 1, 2), (3, 1, 2), args.num_cycles
    )

    env = BBCodeDecodingEnv(
        m=args.m, ell=args.ell,
        num_cycles=args.num_cycles,
        error_rate=args.error_rate
    )

    # Optional: Analyze BP-OSD panel statistics
    if args.analyze_bposd:
        print("\n" + "=" * 70)
        print("ANALYZING BP-OSD PANEL STATISTICS")
        print("=" * 70)
        analyze_bposd_panel_statistics(
            code_data,
            num_samples=1000,
            error_rate=args.error_rate
        )

    # Train according to strategy
    decoders = {}

    if args.strategy == 'scratch':
        decoder, path = train_from_scratch(args, save_path)
        decoders['from_scratch'] = decoder

    elif args.strategy == 'supervised_only':
        decoder, path = train_supervised_only(args, code_data, env, save_path)
        decoders['supervised_only'] = decoder

    elif args.strategy == 'mixed':
        decoder, path = train_mixed(args, code_data, env, save_path)
        decoders['mixed'] = decoder

    elif args.strategy == 'compare_all':
        print("\n>>> TRAINING ALL THREE STRATEGIES FOR COMPARISON <<<\n")

        # From scratch
        decoder_scratch, _ = train_from_scratch(args, save_path)
        decoders['from_scratch'] = decoder_scratch

        # Supervised only
        decoder_supervised, _ = train_supervised_only(args, code_data, env, save_path)
        decoders['supervised_only'] = decoder_supervised

        # Mixed
        decoder_mixed, _ = train_mixed(args, code_data, env, save_path)
        decoders['mixed'] = decoder_mixed

    # Evaluate and compare
    if decoders:
        comparison = evaluate_and_compare(decoders, args, save_path)

        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL COMPARISON SUMMARY")
        print("=" * 70)

        for name, results in comparison.items():
            ler = results['logical_error_rates'][1]  # Middle error rate
            success = results['success_rates'][1]
            time = results['avg_decode_time'][1] * 1000

            print(f"\n{name}:")
            print(f"  LER @ p={args.error_rate:.4f}: {ler:.4f}")
            print(f"  Success rate: {success:.4f}")
            print(f"  Decode time: {time:.2f} ms")

        # Determine winner
        lers = {name: results['logical_error_rates'][1]
                for name, results in comparison.items()}
        best = min(lers, key=lers.get)
        print(f"\n{'=' * 70}")
        print(f"BEST PERFORMER: {best} (LER = {lers[best]:.4f})")
        print(f"{'=' * 70}")

        # Compare mixed vs scratch
        if 'mixed' in lers and 'from_scratch' in lers:
            improvement = (lers['from_scratch'] - lers['mixed']) / lers['from_scratch'] * 100
            print(f"\nMixed training improvement over scratch: {improvement:.1f}%")

    print(f"\n✓ All results saved to: {save_path}")


if __name__ == '__main__':
    main()
