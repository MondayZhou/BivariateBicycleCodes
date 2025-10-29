"""
Visualization Tools for Two-Agent RL Decoder

Provides functions to visualize:
1. Training curves (rewards, losses, success rates)
2. Decoding process (syndrome evolution, corrections)
3. Agent communication patterns
4. Comparison plots between decoders
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import json


class TrainingVisualizer:
    """Visualize training metrics."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.metrics = {
            'rewards': [],
            'success_rates': [],
            'logical_error_rates': [],
            'policy_loss_left': [],
            'policy_loss_right': [],
            'value_loss_left': [],
            'value_loss_right': [],
            'entropy_left': [],
            'entropy_right': []
        }

    def add_metrics(self, metrics: Dict):
        """Add metrics from training step."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot comprehensive training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Two-Agent RL Decoder Training Metrics', fontsize=16)

        # Rewards
        if self.metrics['rewards']:
            axes[0, 0].plot(self.metrics['rewards'], alpha=0.6)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Update')
            axes[0, 0].set_ylabel('Mean Reward')
            axes[0, 0].grid(True, alpha=0.3)

        # Success rates
        if self.metrics['success_rates']:
            axes[0, 1].plot(self.metrics['success_rates'], color='green', alpha=0.8)
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Update')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].grid(True, alpha=0.3)

        # Logical error rates
        if self.metrics['logical_error_rates']:
            axes[0, 2].plot(self.metrics['logical_error_rates'], color='red', alpha=0.8)
            axes[0, 2].set_title('Logical Error Rate')
            axes[0, 2].set_xlabel('Update')
            axes[0, 2].set_ylabel('Logical Error Rate')
            axes[0, 2].set_ylim([0, 1])
            axes[0, 2].grid(True, alpha=0.3)

        # Policy losses
        if self.metrics['policy_loss_left'] and self.metrics['policy_loss_right']:
            axes[1, 0].plot(self.metrics['policy_loss_left'], label='Left Agent', alpha=0.8)
            axes[1, 0].plot(self.metrics['policy_loss_right'], label='Right Agent', alpha=0.8)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Value losses
        if self.metrics['value_loss_left'] and self.metrics['value_loss_right']:
            axes[1, 1].plot(self.metrics['value_loss_left'], label='Left Agent', alpha=0.8)
            axes[1, 1].plot(self.metrics['value_loss_right'], label='Right Agent', alpha=0.8)
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Entropy
        if self.metrics['entropy_left'] and self.metrics['entropy_right']:
            axes[1, 2].plot(self.metrics['entropy_left'], label='Left Agent', alpha=0.8)
            axes[1, 2].plot(self.metrics['entropy_right'], label='Right Agent', alpha=0.8)
            axes[1, 2].set_title('Policy Entropy')
            axes[1, 2].set_xlabel('Update')
            axes[1, 2].set_ylabel('Entropy')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")

        plt.show()


class DecoderComparison:
    """Visualize comparison between different decoders."""

    def __init__(self):
        pass

    def plot_error_rate_comparison(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot logical error rate vs physical error rate for multiple decoders.

        Args:
            results: Dictionary mapping decoder names to their results
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = {
            'bposd': 'blue',
            'two_agent': 'red',
            'single_agent': 'green',
            'hybrid': 'purple'
        }

        markers = {
            'bposd': 'o',
            'two_agent': 's',
            'single_agent': '^',
            'hybrid': 'D'
        }

        # Logical Error Rate
        for decoder_name, data in results.items():
            error_rates = data['error_rates']
            ler = data['logical_error_rates']

            axes[0].plot(
                error_rates,
                ler,
                marker=markers.get(decoder_name, 'o'),
                color=colors.get(decoder_name, 'black'),
                label=decoder_name,
                linewidth=2,
                markersize=8
            )

        axes[0].set_xlabel('Physical Error Rate', fontsize=12)
        axes[0].set_ylabel('Logical Error Rate', fontsize=12)
        axes[0].set_title('Decoder Comparison: Logical Error Rate', fontsize=14)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, which='both')

        # Decode Time
        for decoder_name, data in results.items():
            error_rates = data['error_rates']
            decode_time = np.array(data['avg_decode_time']) * 1000  # Convert to ms

            axes[1].plot(
                error_rates,
                decode_time,
                marker=markers.get(decoder_name, 'o'),
                color=colors.get(decoder_name, 'black'),
                label=decoder_name,
                linewidth=2,
                markersize=8
            )

        axes[1].set_xlabel('Physical Error Rate', fontsize=12)
        axes[1].set_ylabel('Average Decode Time (ms)', fontsize=12)
        axes[1].set_title('Decoder Comparison: Decode Time', fontsize=14)
        axes[1].set_xscale('log')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")

        plt.show()

    def plot_success_rate_comparison(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """Plot success rate comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            'bposd': 'blue',
            'two_agent': 'red',
            'single_agent': 'green',
            'hybrid': 'purple'
        }

        markers = {
            'bposd': 'o',
            'two_agent': 's',
            'single_agent': '^',
            'hybrid': 'D'
        }

        for decoder_name, data in results.items():
            error_rates = data['error_rates']
            success_rates = data['success_rates']

            ax.plot(
                error_rates,
                success_rates,
                marker=markers.get(decoder_name, 'o'),
                color=colors.get(decoder_name, 'black'),
                label=decoder_name,
                linewidth=2,
                markersize=8
            )

        ax.set_xlabel('Physical Error Rate', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Decoder Comparison: Success Rate', fontsize=14)
        ax.set_xscale('log')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Success rate comparison saved to {save_path}")

        plt.show()


class DecodingVisualizer:
    """Visualize the decoding process step by step."""

    def __init__(self, m: int, ell: int):
        self.m = m
        self.ell = ell

    def plot_syndrome_evolution(
        self,
        syndrome_history: List[np.ndarray],
        save_path: Optional[str] = None
    ):
        """Plot how syndrome evolves during decoding."""
        num_steps = len(syndrome_history)

        fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))

        if num_steps == 1:
            axes = [axes]

        for step, syndrome in enumerate(syndrome_history):
            # Reshape syndrome to 2D grid for visualization
            syndrome_2d = syndrome.reshape(self.m, self.ell)

            axes[step].imshow(syndrome_2d, cmap='RdBu', vmin=0, vmax=1)
            axes[step].set_title(f'Step {step}')
            axes[step].set_xlabel('ell')
            axes[step].set_ylabel('m')

            # Add grid
            for i in range(self.m + 1):
                axes[step].axhline(i - 0.5, color='gray', linewidth=0.5)
            for j in range(self.ell + 1):
                axes[step].axvline(j - 0.5, color='gray', linewidth=0.5)

        plt.suptitle('Syndrome Evolution During Decoding', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_correction_pattern(
        self,
        correction_left: np.ndarray,
        correction_right: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Visualize correction pattern on both panels."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left panel
        correction_left_2d = correction_left.reshape(self.m, self.ell)
        im1 = axes[0].imshow(correction_left_2d, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[0].set_title('Left Panel Correction', fontsize=12)
        axes[0].set_xlabel('ell')
        axes[0].set_ylabel('m')
        plt.colorbar(im1, ax=axes[0])

        # Right panel
        correction_right_2d = correction_right.reshape(self.m, self.ell)
        im2 = axes[1].imshow(correction_right_2d, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1].set_title('Right Panel Correction', fontsize=12)
        axes[1].set_xlabel('ell')
        axes[1].set_ylabel('m')
        plt.colorbar(im2, ax=axes[1])

        plt.suptitle('Two-Agent Decoder Corrections', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


def load_results(filename: str) -> Dict:
    """Load experimental results from JSON file."""
    with open(filename, 'r') as f:
        results = json.load(f)
    return results


def create_summary_report(
    results: Dict[str, Dict],
    save_path: str = 'decoder_comparison_report.txt'
):
    """Create a text summary report of decoder comparison."""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DECODER COMPARISON SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        error_rates = results[list(results.keys())[0]]['error_rates']

        for i, p in enumerate(error_rates):
            f.write(f"\nPhysical Error Rate: p = {p:.4f}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Decoder':<20} {'LER':<15} {'Success Rate':<15} {'Decode Time (ms)':<20}\n")
            f.write("-" * 80 + "\n")

            for decoder_name, data in results.items():
                ler = data['logical_error_rates'][i]
                success = data['success_rates'][i]
                time_ms = data['avg_decode_time'][i] * 1000

                f.write(f"{decoder_name:<20} {ler:<15.6f} {success:<15.4f} {time_ms:<20.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report saved to {save_path}")
