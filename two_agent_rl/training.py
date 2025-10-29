"""
Multi-Agent PPO Training for Two-Agent BB Code Decoder

Implements Proximal Policy Optimization for cooperative two-agent RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import time

from agent_architecture import TwoAgentDecoder
from environment import BBCodeDecodingEnv


class RolloutBuffer:
    """
    Buffer for storing trajectories during rollout.
    """

    def __init__(self):
        self.states_left = []
        self.states_right = []
        self.actions_left = []
        self.actions_right = []
        self.log_probs_left = []
        self.log_probs_right = []
        self.rewards = []
        self.values_left = []
        self.values_right = []
        self.dones = []

    def add(
        self,
        state_left: Dict,
        state_right: Dict,
        action_left: torch.Tensor,
        action_right: torch.Tensor,
        log_prob_left: torch.Tensor,
        log_prob_right: torch.Tensor,
        reward: float,
        value_left: torch.Tensor,
        value_right: torch.Tensor,
        done: bool
    ):
        self.states_left.append(state_left)
        self.states_right.append(state_right)
        self.actions_left.append(action_left)
        self.actions_right.append(action_right)
        self.log_probs_left.append(log_prob_left)
        self.log_probs_right.append(log_prob_right)
        self.rewards.append(reward)
        self.values_left.append(value_left)
        self.values_right.append(value_right)
        self.dones.append(done)

    def clear(self):
        self.states_left = []
        self.states_right = []
        self.actions_left = []
        self.actions_right = []
        self.log_probs_left = []
        self.log_probs_right = []
        self.rewards = []
        self.values_left = []
        self.values_right = []
        self.dones = []

    def __len__(self):
        return len(self.rewards)


class MultiAgentPPO:
    """
    Multi-Agent Proximal Policy Optimization trainer.

    Trains two cooperative agents to jointly decode BB codes.
    """

    def __init__(
        self,
        env: BBCodeDecodingEnv,
        decoder: TwoAgentDecoder,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        device: str = 'cpu'
    ):
        self.env = env
        self.decoder = decoder.to(device)
        self.device = device

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        # Optimizers for both agents
        self.optimizer_left = optim.Adam(decoder.agent_left.parameters(), lr=lr)
        self.optimizer_right = optim.Adam(decoder.agent_right.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Metrics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        self.logical_error_rate = deque(maxlen=100)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0

        # Convert values to tensor
        values = torch.cat(values).squeeze()
        next_value = next_value.squeeze()

        # Backward computation of GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value_t = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def collect_rollouts(self, num_episodes: int) -> Dict:
        """
        Collect rollout data from environment.

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            metrics: Dictionary of metrics from rollout
        """
        self.buffer.clear()

        episode_rewards_batch = []
        episode_lengths_batch = []
        success_count = 0
        logical_error_count = 0

        for episode in range(num_episodes):
            state_left, state_right = self.env.reset()

            # Move states to device
            state_left = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in state_left.items()}
            state_right = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in state_right.items()}

            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Get actions from both agents
                with torch.no_grad():
                    actions_left, log_probs_left, value_left = self.decoder.agent_left.get_action_and_log_prob(
                        state_left['node_features'],
                        state_left['edge_index'],
                        partner_encoding=None,
                        deterministic=False
                    )

                    # Get encoding from left agent for communication
                    _, _, encoding_left = self.decoder.agent_left(
                        state_left['node_features'],
                        state_left['edge_index'],
                        partner_encoding=None
                    )

                    actions_right, log_probs_right, value_right = self.decoder.agent_right.get_action_and_log_prob(
                        state_right['node_features'],
                        state_right['edge_index'],
                        partner_encoding=encoding_left,
                        deterministic=False
                    )

                # Convert actions to numpy
                actions_left_np = actions_left.cpu().numpy()
                actions_right_np = actions_right.cpu().numpy()

                # Environment step
                next_state_left, next_state_right, reward_left, reward_right, done, info = self.env.step(
                    actions_left_np,
                    actions_right_np
                )

                # Use shared reward (cooperative setting)
                reward = (reward_left + reward_right) / 2

                # Store transition
                self.buffer.add(
                    state_left,
                    state_right,
                    actions_left,
                    actions_right,
                    log_probs_left,
                    log_probs_right,
                    reward,
                    value_left,
                    value_right,
                    done
                )

                # Update state
                state_left = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in next_state_left.items()}
                state_right = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in next_state_right.items()}

                episode_reward += reward
                episode_length += 1

            # Episode finished
            episode_rewards_batch.append(episode_reward)
            episode_lengths_batch.append(episode_length)

            if info.get('syndrome_satisfied', False) and not info.get('logical_error', True):
                success_count += 1

            if info.get('logical_error', False):
                logical_error_count += 1

        # Store metrics
        self.episode_rewards.extend(episode_rewards_batch)
        self.episode_lengths.extend(episode_lengths_batch)
        self.success_rate.append(success_count / num_episodes)
        self.logical_error_rate.append(logical_error_count / num_episodes)

        metrics = {
            'mean_reward': np.mean(episode_rewards_batch),
            'mean_length': np.mean(episode_lengths_batch),
            'success_rate': success_count / num_episodes,
            'logical_error_rate': logical_error_count / num_episodes
        }

        return metrics

    def update(self) -> Dict:
        """
        Update both agents using PPO.

        Returns:
            losses: Dictionary of loss values
        """
        # Compute advantages and returns for both agents
        with torch.no_grad():
            # Get next value estimate (last state)
            last_state_left = self.buffer.states_left[-1]
            last_state_right = self.buffer.states_right[-1]

            _, next_value_left, _ = self.decoder.agent_left(
                last_state_left['node_features'],
                last_state_left['edge_index']
            )

            _, next_value_right, _ = self.decoder.agent_right(
                last_state_right['node_features'],
                last_state_right['edge_index']
            )

        # Compute GAE for both agents
        advantages_left, returns_left = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values_left,
            self.buffer.dones,
            next_value_left
        )

        advantages_right, returns_right = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values_right,
            self.buffer.dones,
            next_value_right
        )

        # Convert buffer to tensors
        old_log_probs_left = torch.stack(self.buffer.log_probs_left).to(self.device)
        old_log_probs_right = torch.stack(self.buffer.log_probs_right).to(self.device)
        actions_left = torch.stack(self.buffer.actions_left).to(self.device)
        actions_right = torch.stack(self.buffer.actions_right).to(self.device)

        # PPO update for multiple epochs
        total_policy_loss_left = 0
        total_value_loss_left = 0
        total_policy_loss_right = 0
        total_value_loss_right = 0
        total_entropy_left = 0
        total_entropy_right = 0

        for epoch in range(self.ppo_epochs):
            # Update left agent
            for t in range(len(self.buffer)):
                state_left = self.buffer.states_left[t]

                # Forward pass
                logits, value, _ = self.decoder.agent_left(
                    state_left['node_features'],
                    state_left['edge_index']
                )

                # Compute log probs and entropy
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions_left[t])
                entropy = dist.entropy()

                # Policy loss
                ratio = torch.exp(log_probs - old_log_probs_left[t])
                surr1 = ratio * advantages_left[t]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_left[t]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(value.squeeze(), returns_left[t])

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                # Backward pass
                self.optimizer_left.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.agent_left.parameters(), self.max_grad_norm)
                self.optimizer_left.step()

                total_policy_loss_left += policy_loss.item()
                total_value_loss_left += value_loss.item()
                total_entropy_left += entropy.mean().item()

            # Update right agent
            for t in range(len(self.buffer)):
                state_right = self.buffer.states_right[t]

                # Get encoding from left agent (for communication)
                with torch.no_grad():
                    _, _, encoding_left = self.decoder.agent_left(
                        self.buffer.states_left[t]['node_features'],
                        self.buffer.states_left[t]['edge_index']
                    )

                # Forward pass
                logits, value, _ = self.decoder.agent_right(
                    state_right['node_features'],
                    state_right['edge_index'],
                    partner_encoding=encoding_left
                )

                # Compute log probs and entropy
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions_right[t])
                entropy = dist.entropy()

                # Policy loss
                ratio = torch.exp(log_probs - old_log_probs_right[t])
                surr1 = ratio * advantages_right[t]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_right[t]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(value.squeeze(), returns_right[t])

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                # Backward pass
                self.optimizer_right.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.agent_right.parameters(), self.max_grad_norm)
                self.optimizer_right.step()

                total_policy_loss_right += policy_loss.item()
                total_value_loss_right += value_loss.item()
                total_entropy_right += entropy.mean().item()

        # Average losses
        num_updates = self.ppo_epochs * len(self.buffer)
        losses = {
            'policy_loss_left': total_policy_loss_left / num_updates,
            'value_loss_left': total_value_loss_left / num_updates,
            'entropy_left': total_entropy_left / num_updates,
            'policy_loss_right': total_policy_loss_right / num_updates,
            'value_loss_right': total_value_loss_right / num_updates,
            'entropy_right': total_entropy_right / num_updates,
        }

        return losses

    def train(
        self,
        total_timesteps: int,
        episodes_per_update: int = 10,
        log_interval: int = 10,
        save_interval: int = 100,
        save_path: str = './checkpoints'
    ):
        """
        Main training loop.

        Args:
            total_timesteps: Total number of environment steps to train
            episodes_per_update: Number of episodes to collect before each update
            log_interval: Frequency of logging (in updates)
            save_interval: Frequency of saving checkpoints (in updates)
            save_path: Directory to save checkpoints
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        timesteps_so_far = 0
        update_count = 0

        print("Starting Multi-Agent PPO Training...")
        print(f"Target timesteps: {total_timesteps}")
        print(f"Episodes per update: {episodes_per_update}")

        start_time = time.time()

        while timesteps_so_far < total_timesteps:
            # Collect rollouts
            metrics = self.collect_rollouts(episodes_per_update)
            timesteps_so_far += sum(self.buffer.dones)

            # Update agents
            losses = self.update()

            update_count += 1

            # Logging
            if update_count % log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = timesteps_so_far / elapsed_time

                print(f"\n--- Update {update_count} ---")
                print(f"Timesteps: {timesteps_so_far}/{total_timesteps}")
                print(f"FPS: {fps:.2f}")
                print(f"Mean reward: {metrics['mean_reward']:.2f}")
                print(f"Mean episode length: {metrics['mean_length']:.2f}")
                print(f"Success rate: {metrics['success_rate']:.2%}")
                print(f"Logical error rate: {metrics['logical_error_rate']:.2%}")
                print(f"Policy loss (L/R): {losses['policy_loss_left']:.4f} / {losses['policy_loss_right']:.4f}")
                print(f"Value loss (L/R): {losses['value_loss_left']:.4f} / {losses['value_loss_right']:.4f}")
                print(f"Entropy (L/R): {losses['entropy_left']:.4f} / {losses['entropy_right']:.4f}")

            # Save checkpoint
            if update_count % save_interval == 0:
                checkpoint_path = os.path.join(save_path, f'checkpoint_{update_count}.pt')
                self.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        print("\nTraining complete!")
        final_path = os.path.join(save_path, 'final_model.pt')
        self.save(final_path)
        print(f"Saved final model to {final_path}")

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'agent_left_state_dict': self.decoder.agent_left.state_dict(),
            'agent_right_state_dict': self.decoder.agent_right.state_dict(),
            'optimizer_left_state_dict': self.optimizer_left.state_dict(),
            'optimizer_right_state_dict': self.optimizer_right.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'success_rate': list(self.success_rate),
            'logical_error_rate': list(self.logical_error_rate),
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.decoder.agent_left.load_state_dict(checkpoint['agent_left_state_dict'])
        self.decoder.agent_right.load_state_dict(checkpoint['agent_right_state_dict'])
        self.optimizer_left.load_state_dict(checkpoint['optimizer_left_state_dict'])
        self.optimizer_right.load_state_dict(checkpoint['optimizer_right_state_dict'])
        print(f"Loaded checkpoint from {path}")
