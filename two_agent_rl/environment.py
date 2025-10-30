"""
Multi-Agent Environment for BB Code Decoding

Wraps the existing decoder simulation into a gym-like environment
for training two-agent RL decoders.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
import sys
import os

# Add parent directory to path to import decoder modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoder_setup import bivariate_bicycle_codes
from decoder_run import circuit_simulation


class BBCodeDecodingEnv:
    """
    Environment for two-agent RL decoding of Bivariate Bicycle codes.

    State: Syndrome measurements + qubit features
    Action: Binary correction vector for each panel
    Reward: Based on syndrome reduction and final logical error
    """

    def __init__(
        self,
        m: int = 6,
        ell: int = 12,
        a: Tuple[int, int, int] = (3, 1, 2),
        b: Tuple[int, int, int] = (3, 1, 2),
        num_cycles: int = 3,
        error_rate: float = 0.001,
        reward_type: str = 'mixed',  # 'sparse', 'dense', 'mixed'
        max_steps: int = 5,
        decode_X_errors: bool = True,
        decode_Z_errors: bool = True
    ):
        """
        Initialize BB code decoding environment.

        Args:
            m, ell: BB code parameters
            a, b: Polynomial coefficients
            num_cycles: Number of syndrome measurement cycles
            error_rate: Physical error rate
            reward_type: Type of reward shaping
            max_steps: Maximum decoding steps per episode
            decode_X_errors: Whether to decode X errors
            decode_Z_errors: Whether to decode Z errors
        """
        self.m = m
        self.ell = ell
        self.a = a
        self.b = b
        self.num_cycles = num_cycles
        self.error_rate = error_rate
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.decode_X_errors = decode_X_errors
        self.decode_Z_errors = decode_Z_errors

        # Initialize BB code
        self.code_data = bivariate_bicycle_codes(m, ell, a, b, num_cycles, error_rate)

        # Extract code parameters
        self.num_data_qubits_per_panel = m * ell
        self.num_checks = m * ell
        self.num_total_qubits = 2 * m * ell  # Left + right panels

        # Build Tanner graphs for both panels
        self._build_tanner_graphs()

        # Episode state
        self.current_step = 0
        self.current_syndrome = None
        self.current_error_X = None
        self.current_error_Z = None
        self.correction_left = None
        self.correction_right = None
        self.initial_syndrome_weight = 0

    def _build_tanner_graphs(self):
        """
        Build Tanner graph edge indices for left and right panels.

        Nodes: [data_qubits (0 to m*ell-1), X_checks (m*ell to 2*m*ell-1), Z_checks (2*m*ell to 3*m*ell-1)]
        Edges: Connect checks to data qubits according to neighbor structure
        """
        nbs = self.code_data['nbs']
        num_data = self.num_data_qubits_per_panel
        num_checks = self.num_checks

        # Left panel: neighbors 0, 1, 2 of each check
        # Right panel: neighbors 3, 4, 5 of each check

        edges_left = []
        edges_right = []

        # X-checks (neighbors from nbs dictionary)
        for check_idx in range(num_checks):
            check_node_id = num_data + check_idx  # Offset by data qubits

            # Left panel connections (neighbors 0, 1, 2)
            for nb_idx in range(3):
                data_qubit = nbs[f'x{check_idx}'][nb_idx]
                # Add bidirectional edge
                edges_left.append([check_node_id, data_qubit])
                edges_left.append([data_qubit, check_node_id])

            # Right panel connections (neighbors 3, 4, 5)
            for nb_idx in range(3, 6):
                data_qubit = nbs[f'x{check_idx}'][nb_idx] - num_data  # Remap to [0, m*ell-1]
                edges_right.append([check_node_id, data_qubit])
                edges_right.append([data_qubit, check_node_id])

        # Z-checks
        for check_idx in range(num_checks):
            check_node_id = num_data + num_checks + check_idx  # Offset by data + X-checks

            # Left panel connections (neighbors 0, 1, 2)
            for nb_idx in range(3):
                data_qubit = nbs[f'z{check_idx}'][nb_idx]
                edges_left.append([check_node_id, data_qubit])
                edges_left.append([data_qubit, check_node_id])

            # Right panel connections (neighbors 3, 4, 5)
            for nb_idx in range(3, 6):
                data_qubit = nbs[f'z{check_idx}'][nb_idx] - num_data
                edges_right.append([check_node_id, data_qubit])
                edges_right.append([data_qubit, check_node_id])

        # Convert to torch tensors
        self.edge_index_left = torch.tensor(edges_left, dtype=torch.long).t().contiguous()
        self.edge_index_right = torch.tensor(edges_right, dtype=torch.long).t().contiguous()

        # Total nodes per panel graph: data_qubits + X_checks + Z_checks
        self.num_nodes_per_panel = num_data + 2 * num_checks

    def _create_node_features(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray,
        correction_left: np.ndarray,
        correction_right: np.ndarray,
        panel: str
    ) -> torch.Tensor:
        """
        Create node feature vectors for GNN input.

        Features per node:
        - For data qubits: [is_data, is_X_check, is_Z_check, current_correction, step_number, 0, 0, 0, 0, 0]
        - For X-checks: [is_data, is_X_check, is_Z_check, 0, step_number, syndrome_value, syndrome_weight, 0, 0, 0]
        - For Z-checks: [is_data, is_X_check, is_Z_check, 0, step_number, 0, 0, syndrome_value, syndrome_weight, 0]

        Args:
            syndrome_X: X-check syndromes [num_checks]
            syndrome_Z: Z-check syndromes [num_checks]
            correction_left: Current correction on left panel
            correction_right: Current correction on right panel
            panel: 'left' or 'right'

        Returns:
            node_features: [num_nodes, 10] tensor
        """
        num_data = self.num_data_qubits_per_panel
        num_checks = self.num_checks
        num_nodes = self.num_nodes_per_panel
        feature_dim = 10

        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

        correction = correction_left if panel == 'left' else correction_right

        # Data qubit features (nodes 0 to num_data-1)
        features[:num_data, 0] = 1.0  # is_data
        features[:num_data, 3] = correction  # current_correction
        features[:num_data, 4] = self.current_step / self.max_steps  # Normalized step

        # X-check features (nodes num_data to num_data + num_checks - 1)
        x_check_start = num_data
        x_check_end = num_data + num_checks
        features[x_check_start:x_check_end, 1] = 1.0  # is_X_check
        features[x_check_start:x_check_end, 4] = self.current_step / self.max_steps
        features[x_check_start:x_check_end, 5] = syndrome_X  # syndrome_value
        features[x_check_start:x_check_end, 6] = np.sum(syndrome_X) / num_checks  # syndrome_weight

        # Z-check features (nodes num_data + num_checks to end)
        z_check_start = num_data + num_checks
        z_check_end = num_nodes
        features[z_check_start:z_check_end, 2] = 1.0  # is_Z_check
        features[z_check_start:z_check_end, 4] = self.current_step / self.max_steps
        features[z_check_start:z_check_end, 7] = syndrome_Z  # syndrome_value
        features[z_check_start:z_check_end, 8] = np.sum(syndrome_Z) / num_checks  # syndrome_weight

        return torch.tensor(features, dtype=torch.float32)

    def reset(self) -> Tuple[Dict, Dict]:
        """
        Reset environment and return initial states for both agents.

        Returns:
            state_left: State dict for left panel agent
            state_right: State dict for right panel agent
        """
        # Generate random error
        self.current_error_X, self.current_error_Z, syndrome_dict = circuit_simulation(
            self.code_data,
            self.error_rate
        )

        # Extract syndromes
        syndrome_X = syndrome_dict['X_checks']
        syndrome_Z = syndrome_dict['Z_checks']

        self.current_syndrome = {
            'X_checks': syndrome_X.copy(),
            'Z_checks': syndrome_Z.copy()
        }

        # Initialize corrections
        self.correction_left = np.zeros(self.num_data_qubits_per_panel, dtype=np.int8)
        self.correction_right = np.zeros(self.num_data_qubits_per_panel, dtype=np.int8)

        # Reset step counter
        self.current_step = 0

        # Store initial syndrome weight for reward shaping
        self.initial_syndrome_weight = np.sum(syndrome_X) + np.sum(syndrome_Z)

        # Create initial states
        state_left = self._get_state('left')
        state_right = self._get_state('right')

        return state_left, state_right

    def _get_state(self, panel: str) -> Dict[str, torch.Tensor]:
        """Get current state for a panel."""
        syndrome_X = self.current_syndrome['X_checks']
        syndrome_Z = self.current_syndrome['Z_checks']

        node_features = self._create_node_features(
            syndrome_X,
            syndrome_Z,
            self.correction_left,
            self.correction_right,
            panel
        )

        edge_index = self.edge_index_left if panel == 'left' else self.edge_index_right

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'syndrome_X': torch.tensor(syndrome_X, dtype=torch.float32),
            'syndrome_Z': torch.tensor(syndrome_Z, dtype=torch.float32)
        }

    def step(
        self,
        action_left: np.ndarray,
        action_right: np.ndarray
    ) -> Tuple[Dict, Dict, float, float, bool, Dict]:
        """
        Execute one decoding step with actions from both agents.

        Args:
            action_left: Binary correction vector for left panel [num_data_qubits]
            action_right: Binary correction vector for right panel [num_data_qubits]

        Returns:
            state_left: New state for left agent
            state_right: New state for right agent
            reward_left: Reward for left agent
            reward_right: Reward for right agent
            done: Whether episode is finished
            info: Additional information
        """
        # Update corrections (XOR for Pauli corrections)
        self.correction_left = (self.correction_left + action_left) % 2
        self.correction_right = (self.correction_right + action_right) % 2

        # Compute residual syndrome
        residual_syndrome = self._compute_residual_syndrome()

        # Update current syndrome
        self.current_syndrome['X_checks'] = residual_syndrome['X_checks']
        self.current_syndrome['Z_checks'] = residual_syndrome['Z_checks']

        # Increment step
        self.current_step += 1

        # Check if done
        syndrome_satisfied = (
            np.sum(residual_syndrome['X_checks']) == 0 and
            np.sum(residual_syndrome['Z_checks']) == 0
        )

        done = syndrome_satisfied or (self.current_step >= self.max_steps)

        # Compute rewards
        reward_left, reward_right, info = self._compute_rewards(
            residual_syndrome,
            syndrome_satisfied,
            done
        )

        # Get new states
        state_left = self._get_state('left')
        state_right = self._get_state('right')

        return state_left, state_right, reward_left, reward_right, done, info

    def _compute_residual_syndrome(self) -> Dict[str, np.ndarray]:
        """
        Compute residual syndrome after applying current corrections.

        Uses the check matrices from code_data to compute syndrome.
        """
        # Combine corrections from both panels
        full_correction = np.concatenate([self.correction_left, self.correction_right])

        # Get check matrices
        hx = self.code_data['hx']  # X-checks
        hz = self.code_data['hz']  # Z-checks

        # Compute syndrome (matrix-vector product mod 2)
        syndrome_X = (hx @ full_correction) % 2
        syndrome_Z = (hz @ full_correction) % 2

        # XOR with original syndrome from errors
        original_syndrome_X = (hx @ np.concatenate([
            self.current_error_X[:self.num_data_qubits_per_panel],
            self.current_error_X[self.num_data_qubits_per_panel:]
        ])) % 2

        original_syndrome_Z = (hz @ np.concatenate([
            self.current_error_Z[:self.num_data_qubits_per_panel],
            self.current_error_Z[self.num_data_qubits_per_panel:]
        ])) % 2

        residual_X = (original_syndrome_X + syndrome_X) % 2
        residual_Z = (original_syndrome_Z + syndrome_Z) % 2

        return {
            'X_checks': residual_X,
            'Z_checks': residual_Z
        }

    def _compute_rewards(
        self,
        residual_syndrome: Dict[str, np.ndarray],
        syndrome_satisfied: bool,
        done: bool
    ) -> Tuple[float, float, Dict]:
        """
        Compute rewards for both agents.

        Reward types:
        - 'sparse': Only reward on episode completion
        - 'dense': Reward syndrome reduction at each step
        - 'mixed': Combination of both
        """
        syndrome_weight = np.sum(residual_syndrome['X_checks']) + np.sum(residual_syndrome['Z_checks'])

        info = {
            'syndrome_weight': syndrome_weight,
            'syndrome_satisfied': syndrome_satisfied,
            'logical_error': False
        }

        if self.reward_type == 'sparse':
            if done:
                if syndrome_satisfied:
                    # Check for logical error
                    logical_error = self._check_logical_error()
                    info['logical_error'] = logical_error

                    if not logical_error:
                        reward = 100.0  # Success
                    else:
                        reward = -100.0  # Logical error
                else:
                    reward = -50.0  # Failed to satisfy syndrome
            else:
                reward = 0.0

        elif self.reward_type == 'dense':
            # Reward syndrome reduction
            prev_weight = info.get('prev_syndrome_weight', self.initial_syndrome_weight)
            syndrome_reduction = prev_weight - syndrome_weight
            reward = syndrome_reduction * 10.0

            # Penalty for corrections
            num_corrections = np.sum(self.correction_left) + np.sum(self.correction_right)
            reward -= num_corrections * 0.1

            if done and syndrome_satisfied:
                logical_error = self._check_logical_error()
                info['logical_error'] = logical_error
                reward += 100.0 if not logical_error else -100.0

        else:  # 'mixed'
            # Dense reward for syndrome reduction
            prev_weight = info.get('prev_syndrome_weight', self.initial_syndrome_weight)
            syndrome_reduction = prev_weight - syndrome_weight
            reward = syndrome_reduction * 5.0

            # Sparse reward for completion
            if done:
                if syndrome_satisfied:
                    logical_error = self._check_logical_error()
                    info['logical_error'] = logical_error
                    reward += 50.0 if not logical_error else -50.0
                else:
                    reward -= 25.0

        # Both agents get same reward (cooperative)
        reward_left = reward
        reward_right = reward

        info['prev_syndrome_weight'] = syndrome_weight

        return reward_left, reward_right, info

    def _check_logical_error(self) -> bool:
        """
        Check if current correction introduces a logical error.

        Returns True if there is a logical error, False otherwise.
        """
        # Combine total error: original error + correction
        total_error_X = (self.current_error_X + np.concatenate([
            self.correction_left, self.correction_right
        ])) % 2

        total_error_Z = (self.current_error_Z + np.concatenate([
            self.correction_left, self.correction_right
        ])) % 2

        # Check logical operators (from code_data)
        logicOp_X = self.code_data.get('logicOp_X', None)
        logicOp_Z = self.code_data.get('logicOp_Z', None)

        logical_error = False

        if logicOp_X is not None and self.decode_X_errors:
            # Check if total_error_X anticommutes with logical Z
            logical_X_error = np.any((logicOp_X @ total_error_X) % 2 != 0)
            logical_error = logical_error or logical_X_error

        if logicOp_Z is not None and self.decode_Z_errors:
            # Check if total_error_Z anticommutes with logical X
            logical_Z_error = np.any((logicOp_Z @ total_error_Z) % 2 != 0)
            logical_error = logical_error or logical_Z_error

        return logical_error

    def render(self):
        """Print current environment state (for debugging)."""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"X-syndrome weight: {np.sum(self.current_syndrome['X_checks'])}")
        print(f"Z-syndrome weight: {np.sum(self.current_syndrome['Z_checks'])}")
        print(f"Corrections left: {np.sum(self.correction_left)}")
        print(f"Corrections right: {np.sum(self.correction_right)}")
