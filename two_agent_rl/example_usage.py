"""
Example Usage of Two-Agent RL Decoder

This script demonstrates the complete workflow:
1. Train a two-agent decoder
2. Evaluate performance
3. Compare with baselines
4. Use hybrid decoder
"""

import torch
import numpy as np
from agent_architecture import TwoAgentDecoder
from single_agent_baseline import SingleAgentDecoder
from environment import BBCodeDecodingEnv
from training import MultiAgentPPO
from hybrid_decoder import HybridBPOSD_RL_Decoder
from experiments import DecoderBenchmark
from visualize import DecoderComparison, TrainingVisualizer
from decoder_setup import bivariate_bicycle_codes


def example_1_train_two_agent_decoder():
    """Example 1: Train a two-agent decoder from scratch."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Training Two-Agent Decoder")
    print("=" * 60)

    # Create environment
    env = BBCodeDecodingEnv(
        m=6,
        ell=12,
        a=(3, 1, 2),
        b=(3, 1, 2),
        num_cycles=3,
        error_rate=0.001,
        reward_type='mixed',
        max_steps=5
    )

    # Create decoder
    decoder = TwoAgentDecoder(
        m=6,
        ell=12,
        hidden_dim=128,
        num_gnn_layers=4,
        num_attention_heads=4,
        dropout=0.1
    )

    # Create trainer
    trainer = MultiAgentPPO(
        env=env,
        decoder=decoder,
        lr=3e-4,
        gamma=0.99,
        device='cpu'
    )

    # Train (small example - use more timesteps in practice)
    print("\nStarting training...")
    trainer.train(
        total_timesteps=10000,  # Use 100000+ for real training
        episodes_per_update=10,
        log_interval=5,
        save_interval=50,
        save_path='./checkpoints/example'
    )

    print("\nTraining complete!")
    return decoder


def example_2_evaluate_decoder(decoder):
    """Example 2: Evaluate trained decoder."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Evaluating Decoder Performance")
    print("=" * 60)

    # Create benchmark
    benchmark = DecoderBenchmark(m=6, ell=12, device='cpu')

    # Test across multiple error rates
    error_rates = [0.0005, 0.001, 0.002]
    num_trials = 100  # Use 1000+ for real evaluation

    print("\nEvaluating two-agent decoder...")
    results = benchmark.evaluate_decoder(
        decoder,
        decoder_type='two_agent',
        error_rates=error_rates,
        num_trials=num_trials,
        verbose=True
    )

    # Print summary
    print("\n" + "-" * 60)
    print("RESULTS SUMMARY")
    print("-" * 60)
    for i, p in enumerate(error_rates):
        print(f"Error rate p = {p:.4f}:")
        print(f"  Logical Error Rate: {results['logical_error_rates'][i]:.4f}")
        print(f"  Success Rate: {results['success_rates'][i]:.4f}")
        print(f"  Avg Decode Time: {results['avg_decode_time'][i]*1000:.2f} ms")

    return results


def example_3_compare_decoders():
    """Example 3: Compare two-agent vs single-agent vs BP-OSD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Decoder Comparison")
    print("=" * 60)

    # Create decoders
    two_agent_decoder = TwoAgentDecoder(m=6, ell=12)
    single_agent_decoder = SingleAgentDecoder(m=6, ell=12)

    # Note: In practice, load pre-trained models:
    # two_agent_decoder.load_state_dict(torch.load('path/to/checkpoint.pt'))
    # single_agent_decoder.load_state_dict(torch.load('path/to/checkpoint.pt'))

    # Create benchmark
    benchmark = DecoderBenchmark(m=6, ell=12, device='cpu')

    # Run comparison
    error_rates = [0.001]
    num_trials = 50  # Small for demo

    print("\nRunning comparison (this may take a while)...")
    comparison = benchmark.compare_decoders(
        two_agent_decoder=two_agent_decoder,
        single_agent_decoder=single_agent_decoder,
        error_rates=error_rates,
        num_trials=num_trials,
        include_bposd=True
    )

    # Print summary
    benchmark.print_summary(comparison)

    # Save results
    benchmark.save_results(comparison, 'comparison_results.json')

    # Visualize
    vis = DecoderComparison()
    vis.plot_error_rate_comparison(comparison, save_path='comparison.png')

    return comparison


def example_4_hybrid_decoder():
    """Example 4: Use hybrid BP-OSD + RL decoder."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Hybrid Decoder")
    print("=" * 60)

    # Create code
    code_data = bivariate_bicycle_codes(
        m=6,
        ell=12,
        a=(3, 1, 2),
        b=(3, 1, 2),
        num_cycles=3
    )

    # Create RL decoder
    rl_decoder = TwoAgentDecoder(m=6, ell=12)

    # Create hybrid decoder
    hybrid = HybridBPOSD_RL_Decoder(
        code_data=code_data,
        rl_decoder=rl_decoder,
        weight_threshold=18,
        confidence_threshold=0.5,
        use_rl_for_failures=True,
        device='cpu'
    )

    print("\nHybrid decoder created!")
    print("Strategy:")
    print("  1. Try BP-OSD first (fast)")
    print("  2. If high weight or low confidence → switch to RL")
    print("  3. If BP-OSD fails → fallback to RL")

    # Simulate a few decodes
    from decoder_run import circuit_simulation

    print("\nSimulating 10 decoding instances...")
    for i in range(10):
        # Generate error
        error_X, error_Z, syndrome_dict = circuit_simulation(
            code_data,
            error_rate=0.001
        )

        # Decode (simplified - normally would provide full node features)
        syndrome_X = syndrome_dict['X_checks']
        syndrome_Z = syndrome_dict['Z_checks']

        # Note: Full decode would require node features
        # correction_left, correction_right, info = hybrid.decode(...)

        print(f"  Instance {i+1}: Generated error with syndrome weight "
              f"{np.sum(syndrome_X) + np.sum(syndrome_Z)}")

    # Print statistics
    hybrid.print_statistics()

    return hybrid


def example_5_visualize_training():
    """Example 5: Visualize training progress."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Training Visualization")
    print("=" * 60)

    # Create visualizer
    vis = TrainingVisualizer()

    # Simulate some training metrics
    print("\nSimulating training data...")
    for update in range(100):
        metrics = {
            'rewards': np.random.randn() * 10 + 40 + update * 0.5,
            'success_rates': min(0.5 + update * 0.005, 0.95),
            'logical_error_rates': max(0.4 - update * 0.003, 0.05),
            'policy_loss_left': np.exp(-update * 0.01) * 0.5,
            'policy_loss_right': np.exp(-update * 0.01) * 0.5,
            'value_loss_left': np.exp(-update * 0.02) * 1.0,
            'value_loss_right': np.exp(-update * 0.02) * 1.0,
            'entropy_left': 0.7 - update * 0.005,
            'entropy_right': 0.7 - update * 0.005
        }
        vis.add_metrics(metrics)

    # Plot training curves
    print("\nGenerating training curves...")
    vis.plot_training_curves(save_path='training_curves.png')

    print("\nTraining curves saved!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" TWO-AGENT RL DECODER FOR BIVARIATE BICYCLE CODES - EXAMPLES")
    print("=" * 70)

    # Example 1: Train decoder (commented out - takes time)
    # decoder = example_1_train_two_agent_decoder()

    # Example 2: Evaluate decoder (commented out - requires trained model)
    # results = example_2_evaluate_decoder(decoder)

    # Example 3: Compare decoders (commented out - takes time)
    # comparison = example_3_compare_decoders()

    # Example 4: Hybrid decoder (quick demo)
    hybrid = example_4_hybrid_decoder()

    # Example 5: Visualize training (quick demo)
    example_5_visualize_training()

    print("\n" + "=" * 70)
    print(" EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nTo run full experiments:")
    print("  1. Train: python train.py --total_timesteps 100000")
    print("  2. Evaluate: Use experiments.py with trained models")
    print("  3. Visualize: Use visualize.py with saved results")
    print("\nSee README.md for detailed usage instructions.")


if __name__ == '__main__':
    main()
