#!/usr/bin/env python3
"""
Complete Experimental Pipeline for Two-Agent RL Decoder

This script runs the full experimental workflow:
1. Train two-agent decoder
2. Train single-agent decoder (baseline)
3. Evaluate both + BP-OSD + Hybrid
4. Generate comparison plots
5. Save results

Usage:
    # Quick test (small):
    python run_experiments.py --mode quick

    # Full experiment (takes time):
    python run_experiments.py --mode full

    # Only evaluate (requires trained models):
    python run_experiments.py --mode evaluate --load_two_agent ./checkpoints/two_agent.pt
"""

import argparse
import os
import torch
import numpy as np
import json
from datetime import datetime

from environment import BBCodeDecodingEnv
from agent_architecture import TwoAgentDecoder
from single_agent_baseline import SingleAgentDecoder
from training import MultiAgentPPO
from hybrid_decoder import HybridBPOSD_RL_Decoder
from experiments import DecoderBenchmark
from visualize import DecoderComparison, TrainingVisualizer, create_summary_report
from decoder_setup import bivariate_bicycle_codes


def train_two_agent(args, save_path):
    """Train two-agent decoder."""
    print("\n" + "=" * 70)
    print("STEP 1: Training Two-Agent RL Decoder")
    print("=" * 70)

    # Create environment
    env = BBCodeDecodingEnv(
        m=args.m,
        ell=args.ell,
        a=(3, 1, 2),
        b=(3, 1, 2),
        num_cycles=args.num_cycles,
        error_rate=args.train_error_rate,
        reward_type=args.reward_type,
        max_steps=args.max_steps
    )

    # Create decoder
    decoder = TwoAgentDecoder(
        m=args.m,
        ell=args.ell,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout
    )

    # Create trainer
    trainer = MultiAgentPPO(
        env=env,
        decoder=decoder,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device
    )

    # Train
    trainer.train(
        total_timesteps=args.two_agent_timesteps,
        episodes_per_update=args.episodes_per_update,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_path=os.path.join(save_path, 'two_agent')
    )

    # Save final model
    model_path = os.path.join(save_path, 'two_agent_final.pt')
    torch.save({
        'agent_left': decoder.agent_left.state_dict(),
        'agent_right': decoder.agent_right.state_dict(),
        'args': vars(args)
    }, model_path)

    print(f"\nTwo-agent model saved to: {model_path}")
    return decoder, model_path


def train_single_agent(args, save_path):
    """Train single-agent decoder for comparison."""
    print("\n" + "=" * 70)
    print("STEP 2: Training Single-Agent RL Decoder (Baseline)")
    print("=" * 70)

    # Create environment (same as two-agent)
    env = BBCodeDecodingEnv(
        m=args.m,
        ell=args.ell,
        a=(3, 1, 2),
        b=(3, 1, 2),
        num_cycles=args.num_cycles,
        error_rate=args.train_error_rate,
        reward_type=args.reward_type,
        max_steps=args.max_steps
    )

    # Create single-agent decoder
    decoder = SingleAgentDecoder(
        m=args.m,
        ell=args.ell,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout
    )

    # Note: Single-agent training would need a modified trainer
    # For now, we create it but skip training in quick mode
    print("Note: Single-agent training requires SingleAgentPPO trainer (not implemented yet)")
    print("Using untrained single-agent as baseline comparison point")

    model_path = os.path.join(save_path, 'single_agent_final.pt')
    torch.save({
        'agent': decoder.agent.state_dict(),
        'args': vars(args)
    }, model_path)

    print(f"\nSingle-agent model saved to: {model_path}")
    return decoder, model_path


def evaluate_decoders(args, two_agent_decoder, single_agent_decoder, save_path):
    """Evaluate all decoders and compare."""
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating All Decoders")
    print("=" * 70)

    # Create benchmark
    benchmark = DecoderBenchmark(
        m=args.m,
        ell=args.ell,
        device=args.device
    )

    # Define error rates to test
    if args.mode == 'quick':
        error_rates = [0.001, 0.002]
        num_trials = 50
    else:
        error_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005]
        num_trials = args.eval_trials

    print(f"\nTesting error rates: {error_rates}")
    print(f"Trials per rate: {num_trials}")

    # Run comparison
    comparison = benchmark.compare_decoders(
        two_agent_decoder=two_agent_decoder,
        single_agent_decoder=single_agent_decoder if args.include_single_agent else None,
        error_rates=error_rates,
        num_trials=num_trials,
        include_bposd=args.include_bposd
    )

    # Add hybrid decoder if requested
    if args.include_hybrid and two_agent_decoder is not None:
        print("\n" + "=" * 70)
        print("STEP 3b: Evaluating Hybrid Decoder")
        print("=" * 70)

        code_data = bivariate_bicycle_codes(
            args.m, args.ell, (3, 1, 2), (3, 1, 2), args.num_cycles
        )

        hybrid_decoder = HybridBPOSD_RL_Decoder(
            code_data=code_data,
            rl_decoder=two_agent_decoder,
            weight_threshold=args.m * args.ell // 4,
            confidence_threshold=0.5,
            device=args.device
        )

        comparison['hybrid'] = benchmark.evaluate_decoder(
            hybrid_decoder,
            'hybrid',
            error_rates,
            num_trials,
            verbose=True
        )

        # Print hybrid statistics
        hybrid_decoder.print_statistics()

    # Save results
    results_path = os.path.join(save_path, 'comparison_results.json')
    benchmark.save_results(comparison, results_path)

    # Print summary
    benchmark.print_summary(comparison)

    # Create text report
    report_path = os.path.join(save_path, 'comparison_report.txt')
    create_summary_report(comparison, report_path)

    return comparison


def visualize_results(comparison, save_path):
    """Generate visualization plots."""
    print("\n" + "=" * 70)
    print("STEP 4: Generating Visualizations")
    print("=" * 70)

    vis = DecoderComparison()

    # Error rate comparison
    print("\nGenerating error rate comparison plot...")
    vis.plot_error_rate_comparison(
        comparison,
        save_path=os.path.join(save_path, 'error_rate_comparison.png')
    )

    # Success rate comparison
    print("Generating success rate comparison plot...")
    vis.plot_success_rate_comparison(
        comparison,
        save_path=os.path.join(save_path, 'success_rate_comparison.png')
    )

    print(f"\nVisualizations saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run complete experimental pipeline for two-agent RL decoder'
    )

    # Experiment mode
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'evaluate', 'train_only'],
                        help='Experiment mode: quick (demo), full (complete), evaluate (skip training)')

    # Code parameters
    parser.add_argument('--m', type=int, default=6, help='BB code parameter m')
    parser.add_argument('--ell', type=int, default=12, help='BB code parameter ell')
    parser.add_argument('--num_cycles', type=int, default=3, help='Syndrome cycles')

    # Training parameters
    parser.add_argument('--train_error_rate', type=float, default=0.001,
                        help='Error rate for training')
    parser.add_argument('--two_agent_timesteps', type=int, default=None,
                        help='Training timesteps for two-agent (default: depends on mode)')
    parser.add_argument('--single_agent_timesteps', type=int, default=None,
                        help='Training timesteps for single-agent')

    # Network architecture
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--episodes_per_update', type=int, default=10)
    parser.add_argument('--reward_type', type=str, default='mixed')
    parser.add_argument('--max_steps', type=int, default=5)

    # Evaluation parameters
    parser.add_argument('--eval_trials', type=int, default=1000,
                        help='Number of trials per error rate')
    parser.add_argument('--include_bposd', action='store_true', default=True,
                        help='Include BP-OSD baseline')
    parser.add_argument('--include_single_agent', action='store_true', default=True,
                        help='Include single-agent baseline')
    parser.add_argument('--include_hybrid', action='store_true', default=True,
                        help='Include hybrid decoder')

    # Logging
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='./results',
                        help='Directory for saving results')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])

    # Load pretrained models
    parser.add_argument('--load_two_agent', type=str, default=None,
                        help='Path to pretrained two-agent model')
    parser.add_argument('--load_single_agent', type=str, default=None,
                        help='Path to pretrained single-agent model')

    args = parser.parse_args()

    # Set defaults based on mode
    if args.mode == 'quick':
        args.two_agent_timesteps = args.two_agent_timesteps or 5000
        args.single_agent_timesteps = args.single_agent_timesteps or 5000
        args.eval_trials = 50
    elif args.mode == 'full':
        args.two_agent_timesteps = args.two_agent_timesteps or 100000
        args.single_agent_timesteps = args.single_agent_timesteps or 100000
        args.eval_trials = 1000

    return args


def main():
    args = parse_args()

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(args.save_path, f'experiment_{timestamp}')
    os.makedirs(save_path, exist_ok=True)

    # Save experiment configuration
    config_path = os.path.join(save_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 70)
    print("TWO-AGENT RL DECODER - EXPERIMENTAL PIPELINE")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Code: [[{2*args.m*args.ell}, ?, ?]] (m={args.m}, ell={args.ell})")
    print(f"Device: {args.device}")
    print(f"Save path: {save_path}")
    print("=" * 70)

    # Initialize models
    two_agent_decoder = None
    single_agent_decoder = None

    # PHASE 1: Training (unless evaluate mode)
    if args.mode != 'evaluate':
        if args.load_two_agent is None:
            two_agent_decoder, two_agent_path = train_two_agent(args, save_path)
        else:
            print(f"\nLoading two-agent model from: {args.load_two_agent}")
            two_agent_decoder = TwoAgentDecoder(
                m=args.m, ell=args.ell,
                hidden_dim=args.hidden_dim,
                num_gnn_layers=args.num_gnn_layers
            )
            checkpoint = torch.load(args.load_two_agent, map_location=args.device)
            two_agent_decoder.agent_left.load_state_dict(checkpoint['agent_left'])
            two_agent_decoder.agent_right.load_state_dict(checkpoint['agent_right'])

        if args.include_single_agent and args.load_single_agent is None:
            single_agent_decoder, single_agent_path = train_single_agent(args, save_path)
        elif args.load_single_agent:
            print(f"\nLoading single-agent model from: {args.load_single_agent}")
            single_agent_decoder = SingleAgentDecoder(
                m=args.m, ell=args.ell,
                hidden_dim=args.hidden_dim,
                num_gnn_layers=args.num_gnn_layers
            )
            checkpoint = torch.load(args.load_single_agent, map_location=args.device)
            single_agent_decoder.agent.load_state_dict(checkpoint['agent'])

    # PHASE 2: Evaluation (unless train_only mode)
    if args.mode != 'train_only':
        # Load models if in evaluate mode
        if args.mode == 'evaluate':
            if args.load_two_agent:
                print(f"\nLoading two-agent model from: {args.load_two_agent}")
                two_agent_decoder = TwoAgentDecoder(
                    m=args.m, ell=args.ell,
                    hidden_dim=args.hidden_dim,
                    num_gnn_layers=args.num_gnn_layers
                )
                checkpoint = torch.load(args.load_two_agent, map_location=args.device)
                two_agent_decoder.agent_left.load_state_dict(checkpoint['agent_left'])
                two_agent_decoder.agent_right.load_state_dict(checkpoint['agent_right'])

            if args.load_single_agent:
                print(f"\nLoading single-agent model from: {args.load_single_agent}")
                single_agent_decoder = SingleAgentDecoder(
                    m=args.m, ell=args.ell,
                    hidden_dim=args.hidden_dim,
                    num_gnn_layers=args.num_gnn_layers
                )
                checkpoint = torch.load(args.load_single_agent, map_location=args.device)
                single_agent_decoder.agent.load_state_dict(checkpoint['agent'])

        # Run evaluation
        comparison = evaluate_decoders(
            args,
            two_agent_decoder,
            single_agent_decoder,
            save_path
        )

        # PHASE 3: Visualization
        visualize_results(comparison, save_path)

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {save_path}")
    print("\nGenerated files:")
    print(f"  - config.json (experiment configuration)")
    print(f"  - comparison_results.json (numerical results)")
    print(f"  - comparison_report.txt (text summary)")
    print(f"  - error_rate_comparison.png (LER plot)")
    print(f"  - success_rate_comparison.png (success plot)")
    if args.mode != 'evaluate':
        print(f"  - two_agent_final.pt (trained model)")
        print(f"  - two_agent/checkpoints/* (training checkpoints)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
