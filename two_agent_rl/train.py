"""
Training Script for Two-Agent RL Decoder

Usage:
    python train.py --decoder_type two_agent --total_timesteps 100000
    python train.py --decoder_type single_agent --total_timesteps 100000
"""

import argparse
import torch
import os

from environment import BBCodeDecodingEnv
from agent_architecture import TwoAgentDecoder
from single_agent_baseline import SingleAgentDecoder
from training import MultiAgentPPO


def parse_args():
    parser = argparse.ArgumentParser(description='Train RL decoder for BB codes')

    # Code parameters
    parser.add_argument('--m', type=int, default=6, help='BB code parameter m')
    parser.add_argument('--ell', type=int, default=12, help='BB code parameter ell')
    parser.add_argument('--num_cycles', type=int, default=3, help='Number of syndrome cycles')
    parser.add_argument('--error_rate', type=float, default=0.001, help='Physical error rate')

    # Decoder type
    parser.add_argument('--decoder_type', type=str, default='two_agent',
                        choices=['two_agent', 'single_agent'],
                        help='Type of decoder to train')

    # Network architecture
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_gnn_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--episodes_per_update', type=int, default=10,
                        help='Episodes to collect before each PPO update')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='PPO epochs per update')

    # Environment parameters
    parser.add_argument('--reward_type', type=str, default='mixed',
                        choices=['sparse', 'dense', 'mixed'],
                        help='Reward shaping type')
    parser.add_argument('--max_steps', type=int, default=5,
                        help='Maximum decoding steps per episode')

    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N updates')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save checkpoint every N updates')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for saving')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for training')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.decoder_type}_m{args.m}_ell{args.ell}_p{args.error_rate}"

    save_path = os.path.join(args.save_path, args.experiment_name)

    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Decoder type: {args.decoder_type}")
    print(f"BB code: [[{2 * args.m * args.ell}, ?, ?]]  (m={args.m}, ell={args.ell})")
    print(f"Error rate: {args.error_rate}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Device: {args.device}")
    print(f"Save path: {save_path}")
    print("=" * 60)

    # Create environment
    env = BBCodeDecodingEnv(
        m=args.m,
        ell=args.ell,
        a=(3, 1, 2),
        b=(3, 1, 2),
        num_cycles=args.num_cycles,
        error_rate=args.error_rate,
        reward_type=args.reward_type,
        max_steps=args.max_steps
    )

    # Create decoder
    if args.decoder_type == 'two_agent':
        decoder = TwoAgentDecoder(
            m=args.m,
            ell=args.ell,
            node_feature_dim=10,
            hidden_dim=args.hidden_dim,
            num_gnn_layers=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            dropout=args.dropout
        )
    else:  # single_agent
        decoder = SingleAgentDecoder(
            m=args.m,
            ell=args.ell,
            node_feature_dim=10,
            hidden_dim=args.hidden_dim,
            num_gnn_layers=args.num_gnn_layers,
            num_attention_heads=args.num_attention_heads,
            dropout=args.dropout
        )

    # Create trainer
    if args.decoder_type == 'two_agent':
        trainer = MultiAgentPPO(
            env=env,
            decoder=decoder,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs,
            device=args.device
        )
    else:
        # For single agent, we would need a SingleAgentPPO trainer
        # For simplicity, we can adapt MultiAgentPPO or create a new one
        print("Single agent training not fully implemented yet.")
        print("Please use two_agent decoder type.")
        return

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load(args.resume)

    # Train
    trainer.train(
        total_timesteps=args.total_timesteps,
        episodes_per_update=args.episodes_per_update,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_path=save_path
    )

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
