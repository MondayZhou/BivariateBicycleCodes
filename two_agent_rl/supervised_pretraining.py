"""
Supervised Pretraining from BP-OSD Solutions

This module implements imitation learning where the two-agent RL decoder
learns from BP-OSD solutions before RL fine-tuning.

Key idea: BP-OSD provides "expert demonstrations" that teach the agents
basic decoding patterns, then RL fine-tunes for hard cases.

Benefits:
1. Faster convergence (10-100x fewer samples)
2. Better initialization for independent agents
3. Learns panel-specific features from BP-OSD decomposition
4. Addresses the "cold start" problem in multi-agent RL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoder_setup import bivariate_bicycle_codes
from decoder_run import circuit_simulation
from ldpc.bposd_decoder import bposd_decoder
from agent_architecture import TwoAgentDecoder
from environment import BBCodeDecodingEnv


class BPOSDExpertDataset:
    """
    Generate expert demonstrations from BP-OSD decoder.

    Each sample contains:
    - syndrome (input)
    - BP-OSD correction (label)
    - Panel decomposition (panel-specific labels)
    """

    def __init__(
        self,
        code_data: Dict,
        error_rate: float = 0.001,
        num_samples: int = 10000,
        bp_max_iter: int = 10000,
        only_successful: bool = True
    ):
        """
        Args:
            code_data: BB code data
            error_rate: Physical error rate for generating errors
            num_samples: Number of training samples to generate
            bp_max_iter: Max BP iterations
            only_successful: Only include samples where BP-OSD succeeds
        """
        self.code_data = code_data
        self.error_rate = error_rate
        self.num_samples = num_samples
        self.only_successful = only_successful

        self.m = code_data['m']
        self.ell = code_data['ell']
        self.num_data_per_panel = self.m * self.ell

        # Initialize BP-OSD decoders
        print("Initializing BP-OSD decoders...")
        self.bposd_X = bposd_decoder(
            code_data['hx'],
            error_rate=error_rate,
            max_iter=bp_max_iter,
            bp_method="ms",
            osd_method="osd_cs",
            osd_order=7
        )

        self.bposd_Z = bposd_decoder(
            code_data['hz'],
            error_rate=error_rate,
            max_iter=bp_max_iter,
            bp_method="ms",
            osd_method="osd_cs",
            osd_order=7
        )

        # Storage
        self.samples = []

    def generate_dataset(self, verbose: bool = True):
        """Generate dataset of expert demonstrations."""
        print(f"\nGenerating {self.num_samples} expert demonstrations from BP-OSD...")

        successful = 0
        attempts = 0

        while successful < self.num_samples and attempts < self.num_samples * 3:
            attempts += 1

            # Generate random error
            error_X, error_Z, syndrome_dict = circuit_simulation(
                self.code_data,
                self.error_rate
            )

            syndrome_X = syndrome_dict['X_checks']
            syndrome_Z = syndrome_dict['Z_checks']

            # Get BP-OSD solution
            correction_X = self.bposd_X.decode(syndrome_Z)
            correction_Z = self.bposd_Z.decode(syndrome_X)

            # Check if BP-OSD succeeded (syndrome satisfied)
            total_error = (error_X + error_Z) % 2
            correction_total = (correction_X + correction_Z) % 2
            residual = (total_error + correction_total) % 2

            hx = self.code_data['hx']
            hz = self.code_data['hz']

            residual_syndrome_X = (hx @ residual) % 2
            residual_syndrome_Z = (hz @ residual) % 2

            syndrome_satisfied = (
                np.sum(residual_syndrome_X) == 0 and
                np.sum(residual_syndrome_Z) == 0
            )

            if self.only_successful and not syndrome_satisfied:
                continue

            # Decompose into panels
            correction_left = correction_total[:self.num_data_per_panel]
            correction_right = correction_total[self.num_data_per_panel:]

            # Store sample
            sample = {
                'syndrome_X': syndrome_X.copy(),
                'syndrome_Z': syndrome_Z.copy(),
                'correction_full': correction_total.copy(),
                'correction_left': correction_left.copy(),
                'correction_right': correction_right.copy(),
                'error_X': error_X.copy(),
                'error_Z': error_Z.copy(),
                'success': syndrome_satisfied
            }

            self.samples.append(sample)
            successful += 1

            if verbose and successful % 1000 == 0:
                print(f"  Generated {successful}/{self.num_samples} samples "
                      f"(attempts: {attempts}, success rate: {successful/attempts:.2%})")

        print(f"\nDataset generation complete!")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Success rate: {successful/attempts:.2%}")
        print(f"  Average correction weight: {np.mean([np.sum(s['correction_full']) for s in self.samples]):.2f}")

        return self.samples

    def get_batch(
        self,
        batch_size: int,
        env: BBCodeDecodingEnv
    ) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor]:
        """
        Get a batch of training samples with node features.

        Args:
            batch_size: Number of samples
            env: Environment for generating node features

        Returns:
            states_left: List of states for left agent
            states_right: List of states for right agent
            labels_left: Correction labels for left panel
            labels_right: Correction labels for right panel
        """
        # Sample batch
        indices = np.random.choice(len(self.samples), batch_size, replace=False)
        batch = [self.samples[i] for i in indices]

        states_left = []
        states_right = []
        labels_left = []
        labels_right = []

        for sample in batch:
            # Create node features (similar to environment)
            correction_left_zero = np.zeros(self.num_data_per_panel, dtype=np.int8)
            correction_right_zero = np.zeros(self.num_data_per_panel, dtype=np.int8)

            node_features_left = env._create_node_features(
                sample['syndrome_X'],
                sample['syndrome_Z'],
                correction_left_zero,
                correction_right_zero,
                'left'
            )

            node_features_right = env._create_node_features(
                sample['syndrome_X'],
                sample['syndrome_Z'],
                correction_left_zero,
                correction_right_zero,
                'right'
            )

            states_left.append({
                'node_features': node_features_left,
                'edge_index': env.edge_index_left
            })

            states_right.append({
                'node_features': node_features_right,
                'edge_index': env.edge_index_right
            })

            labels_left.append(torch.tensor(sample['correction_left'], dtype=torch.long))
            labels_right.append(torch.tensor(sample['correction_right'], dtype=torch.long))

        labels_left = torch.stack(labels_left)
        labels_right = torch.stack(labels_right)

        return states_left, states_right, labels_left, labels_right


class SupervisedPretrainer:
    """
    Pretrain two-agent decoder using supervised learning from BP-OSD.
    """

    def __init__(
        self,
        decoder: TwoAgentDecoder,
        dataset: BPOSDExpertDataset,
        env: BBCodeDecodingEnv,
        lr: float = 1e-3,
        device: str = 'cpu'
    ):
        self.decoder = decoder.to(device)
        self.dataset = dataset
        self.env = env
        self.device = device

        # Separate optimizers for each agent
        self.optimizer_left = optim.Adam(decoder.agent_left.parameters(), lr=lr)
        self.optimizer_right = optim.Adam(decoder.agent_right.parameters(), lr=lr)

        # Loss function (cross-entropy for binary classification)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []

    def train_epoch(self, batch_size: int = 32, num_batches: int = 100):
        """Train for one epoch."""
        epoch_loss_left = 0
        epoch_loss_right = 0
        epoch_acc_left = 0
        epoch_acc_right = 0

        for batch_idx in range(num_batches):
            # Get batch
            states_left, states_right, labels_left, labels_right = \
                self.dataset.get_batch(batch_size, self.env)

            labels_left = labels_left.to(self.device)
            labels_right = labels_right.to(self.device)

            # Train left agent
            batch_loss_left = 0
            batch_acc_left = 0

            for i in range(batch_size):
                state_left = {
                    'node_features': states_left[i]['node_features'].to(self.device),
                    'edge_index': states_left[i]['edge_index'].to(self.device)
                }

                # Forward pass
                logits, _, _ = self.decoder.agent_left(
                    state_left['node_features'],
                    state_left['edge_index']
                )

                # Loss
                loss = self.criterion(logits, labels_left[i])

                # Backward pass
                self.optimizer_left.zero_grad()
                loss.backward()
                self.optimizer_left.step()

                batch_loss_left += loss.item()

                # Accuracy
                predictions = torch.argmax(logits, dim=-1)
                acc = (predictions == labels_left[i]).float().mean().item()
                batch_acc_left += acc

            # Train right agent
            batch_loss_right = 0
            batch_acc_right = 0

            for i in range(batch_size):
                state_right = {
                    'node_features': states_right[i]['node_features'].to(self.device),
                    'edge_index': states_right[i]['edge_index'].to(self.device)
                }

                # Get left agent encoding for communication
                with torch.no_grad():
                    _, _, encoding_left = self.decoder.agent_left(
                        states_left[i]['node_features'].to(self.device),
                        states_left[i]['edge_index'].to(self.device)
                    )

                # Forward pass with communication
                logits, _, _ = self.decoder.agent_right(
                    state_right['node_features'],
                    state_right['edge_index'],
                    partner_encoding=encoding_left
                )

                # Loss
                loss = self.criterion(logits, labels_right[i])

                # Backward pass
                self.optimizer_right.zero_grad()
                loss.backward()
                self.optimizer_right.step()

                batch_loss_right += loss.item()

                # Accuracy
                predictions = torch.argmax(logits, dim=-1)
                acc = (predictions == labels_right[i]).float().mean().item()
                batch_acc_right += acc

            epoch_loss_left += batch_loss_left / batch_size
            epoch_loss_right += batch_loss_right / batch_size
            epoch_acc_left += batch_acc_left / batch_size
            epoch_acc_right += batch_acc_right / batch_size

        # Average over batches
        epoch_loss_left /= num_batches
        epoch_loss_right /= num_batches
        epoch_acc_left /= num_batches
        epoch_acc_right /= num_batches

        return {
            'loss_left': epoch_loss_left,
            'loss_right': epoch_loss_right,
            'acc_left': epoch_acc_left,
            'acc_right': epoch_acc_right
        }

    def pretrain(
        self,
        num_epochs: int = 10,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Run supervised pretraining.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            batches_per_epoch: Number of batches per epoch
            save_path: Path to save pretrained model
        """
        print("\n" + "=" * 70)
        print("SUPERVISED PRETRAINING FROM BP-OSD")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {batches_per_epoch}")
        print(f"Dataset size: {len(self.dataset.samples)}")
        print("=" * 70)

        for epoch in range(num_epochs):
            metrics = self.train_epoch(batch_size, batches_per_epoch)

            self.train_losses.append((metrics['loss_left'] + metrics['loss_right']) / 2)
            self.train_accuracies.append((metrics['acc_left'] + metrics['acc_right']) / 2)

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Loss (L/R): {metrics['loss_left']:.4f} / {metrics['loss_right']:.4f}")
            print(f"  Accuracy (L/R): {metrics['acc_left']:.4f} / {metrics['acc_right']:.4f}")

        print("\n" + "=" * 70)
        print("PRETRAINING COMPLETE")
        print("=" * 70)
        print(f"Final accuracy: {self.train_accuracies[-1]:.4f}")

        if save_path:
            torch.save({
                'agent_left': self.decoder.agent_left.state_dict(),
                'agent_right': self.decoder.agent_right.state_dict(),
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies
            }, save_path)
            print(f"Pretrained model saved to: {save_path}")

        return metrics


class MixedTrainer:
    """
    Combined supervised + RL training strategy.

    Phase 1: Supervised pretraining from BP-OSD (warm start)
    Phase 2: RL fine-tuning (learn to handle hard cases)
    """

    def __init__(
        self,
        code_data: Dict,
        decoder: TwoAgentDecoder,
        env: BBCodeDecodingEnv,
        device: str = 'cpu'
    ):
        self.code_data = code_data
        self.decoder = decoder
        self.env = env
        self.device = device

    def train(
        self,
        # Supervised phase
        supervised_samples: int = 10000,
        supervised_epochs: int = 10,
        supervised_lr: float = 1e-3,
        # RL phase
        rl_timesteps: int = 50000,
        rl_lr: float = 3e-4,
        # Paths
        save_dir: str = './checkpoints/mixed_training'
    ):
        """
        Execute mixed training strategy.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("MIXED TRAINING: SUPERVISED PRETRAINING + RL FINE-TUNING")
        print("=" * 70)

        # ===== PHASE 1: SUPERVISED PRETRAINING =====
        print("\n>>> PHASE 1: SUPERVISED PRETRAINING FROM BP-OSD <<<\n")

        # Generate expert dataset
        dataset = BPOSDExpertDataset(
            self.code_data,
            error_rate=self.env.error_rate,
            num_samples=supervised_samples
        )
        dataset.generate_dataset()

        # Pretrain
        pretrainer = SupervisedPretrainer(
            self.decoder,
            dataset,
            self.env,
            lr=supervised_lr,
            device=self.device
        )

        pretrainer.pretrain(
            num_epochs=supervised_epochs,
            save_path=os.path.join(save_dir, 'pretrained.pt')
        )

        # ===== PHASE 2: RL FINE-TUNING =====
        print("\n>>> PHASE 2: RL FINE-TUNING <<<\n")

        from training import MultiAgentPPO

        rl_trainer = MultiAgentPPO(
            env=self.env,
            decoder=self.decoder,
            lr=rl_lr,
            device=self.device
        )

        rl_trainer.train(
            total_timesteps=rl_timesteps,
            episodes_per_update=10,
            log_interval=10,
            save_interval=50,
            save_path=os.path.join(save_dir, 'rl_finetuned')
        )

        # Save final model
        final_path = os.path.join(save_dir, 'final_mixed.pt')
        torch.save({
            'agent_left': self.decoder.agent_left.state_dict(),
            'agent_right': self.decoder.agent_right.state_dict(),
            'supervised_accuracy': pretrainer.train_accuracies[-1],
            'rl_success_rate': list(rl_trainer.success_rate)[-1] if rl_trainer.success_rate else 0
        }, final_path)

        print("\n" + "=" * 70)
        print("MIXED TRAINING COMPLETE")
        print("=" * 70)
        print(f"Supervised accuracy: {pretrainer.train_accuracies[-1]:.4f}")
        if rl_trainer.success_rate:
            print(f"RL success rate: {list(rl_trainer.success_rate)[-1]:.4f}")
        print(f"Final model saved to: {final_path}")
