"""
Single-Agent RL Baseline for Comparison

Implements a single-agent GNN decoder (without panel separation) to compare
against the two-agent approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, Tuple, Optional
import numpy as np


class SingleAgentGNN(nn.Module):
    """
    Single GNN agent that processes the entire Tanner graph without panel separation.

    This serves as a baseline to compare against the two-agent approach.
    """

    def __init__(
        self,
        num_data_qubits: int,
        num_checks: int,
        node_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_gnn_layers: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_data_qubits = num_data_qubits
        self.num_checks = num_checks
        self.hidden_dim = hidden_dim

        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN layers for message passing on full Tanner graph
        self.gnn_layers = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim // num_attention_heads,
                heads=num_attention_heads,
                dropout=dropout,
                concat=True
            )
            for _ in range(num_gnn_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_gnn_layers)
        ])

        # Policy head (actor) - outputs logits for [no_flip, flip] per qubit
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Binary action per qubit
        )

        # Value head (critic) - estimates expected return
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the single agent network.

        Args:
            node_features: [num_nodes, feature_dim] - features for all qubits and checks
            edge_index: [2, num_edges] - full Tanner graph connectivity
            batch: [num_nodes] - batch assignment (optional)

        Returns:
            logits: [num_data_qubits, 2] - action logits for each data qubit
            value: [1] - state value estimate
        """
        # Embed node features
        x = self.node_embedding(node_features)

        # GNN message passing
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            x_new = gnn_layer(x, edge_index)
            x_new = F.relu(x_new)
            x = layer_norm(x + x_new)  # Residual connection

        # Extract data qubit embeddings
        data_qubit_embeddings = x[:self.num_data_qubits]

        # Compute policy logits
        logits = self.policy_net(data_qubit_embeddings)

        # Compute value estimate
        if batch is not None:
            pooled = global_mean_pool(x, batch)
        else:
            pooled = x.mean(dim=0, keepdim=True)
        value = self.value_net(pooled)

        return logits, value

    def get_action_and_log_prob(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and compute log probability.

        Args:
            node_features: Node features
            edge_index: Graph connectivity
            deterministic: If True, take argmax instead of sampling

        Returns:
            actions: [num_data_qubits] - binary actions
            log_probs: [num_data_qubits] - log probabilities of actions
            value: [1] - state value estimate
        """
        logits, value = self.forward(node_features, edge_index)

        # Create categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)

        return actions, log_probs, value


class SingleAgentDecoder(nn.Module):
    """
    Single-agent decoder that processes the entire BB code as one graph.
    """

    def __init__(
        self,
        m: int,
        ell: int,
        node_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_gnn_layers: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.m = m
        self.ell = ell
        self.num_data_qubits = 2 * m * ell  # Both panels combined
        self.num_checks = m * ell

        # Single agent for entire code
        self.agent = SingleAgentGNN(
            num_data_qubits=self.num_data_qubits,
            num_checks=self.num_checks * 2,  # X and Z checks
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        num_iterations: int = 3,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Multi-round decoding with iterative refinement.

        Args:
            state: State dict with 'node_features', 'edge_index'
            num_iterations: Number of iterative refinement rounds
            deterministic: Whether to use deterministic policy

        Returns:
            correction: [num_data_qubits] - full correction
            info: Dict with intermediate information
        """
        correction = torch.zeros(self.num_data_qubits, dtype=torch.long)

        log_probs_list = []
        values_list = []
        actions_list = []

        for iteration in range(num_iterations):
            # Get action from agent
            actions, log_probs, value = self.agent.get_action_and_log_prob(
                state['node_features'],
                state['edge_index'],
                deterministic=deterministic
            )

            # Update correction
            correction ^= actions

            # Store for training
            log_probs_list.append(log_probs)
            values_list.append(value)
            actions_list.append(actions)

        info = {
            'log_probs': torch.stack(log_probs_list),
            'values': torch.stack(values_list),
            'actions': torch.stack(actions_list),
        }

        return correction, info

    def decode(
        self,
        syndrome_dict: Dict[str, np.ndarray],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_iterations: int = 3
    ) -> np.ndarray:
        """
        Inference mode decoding (no gradients, deterministic).

        Args:
            syndrome_dict: Dictionary with 'X_checks' and 'Z_checks'
            node_features: Node features for full graph
            edge_index: Tanner graph for full code
            num_iterations: Number of refinement iterations

        Returns:
            correction: Numpy array of corrections for all qubits
        """
        self.eval()
        with torch.no_grad():
            state = {
                'node_features': node_features,
                'edge_index': edge_index
            }

            correction, _ = self.forward(
                state,
                num_iterations=num_iterations,
                deterministic=True
            )

        return correction.cpu().numpy()


def build_full_tanner_graph(code_data: Dict) -> torch.Tensor:
    """
    Build full Tanner graph for single-agent decoder.

    Combines both panels into a single graph.

    Args:
        code_data: BB code data from decoder_setup

    Returns:
        edge_index: [2, num_edges] tensor
    """
    nbs = code_data['nbs']
    m = code_data['m']
    ell = code_data['ell']
    num_data = 2 * m * ell
    num_checks = m * ell

    edges = []

    # X-checks (all 6 neighbors)
    for check_idx in range(num_checks):
        check_node_id = num_data + check_idx  # Offset by data qubits

        # All 6 neighbors (0-5)
        for nb_idx in range(6):
            data_qubit = nbs[f'x{check_idx}'][nb_idx]
            # Bidirectional edge
            edges.append([check_node_id, data_qubit])
            edges.append([data_qubit, check_node_id])

    # Z-checks (all 6 neighbors)
    for check_idx in range(num_checks):
        check_node_id = num_data + num_checks + check_idx

        # All 6 neighbors (0-5)
        for nb_idx in range(6):
            data_qubit = nbs[f'z{check_idx}'][nb_idx]
            edges.append([check_node_id, data_qubit])
            edges.append([data_qubit, check_node_id])

    # Convert to tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


def create_full_node_features(
    syndrome_X: np.ndarray,
    syndrome_Z: np.ndarray,
    correction: np.ndarray,
    current_step: int,
    max_steps: int,
    num_data_qubits: int,
    num_checks: int
) -> torch.Tensor:
    """
    Create node features for the full graph (single-agent).

    Args:
        syndrome_X: X-check syndrome
        syndrome_Z: Z-check syndrome
        correction: Current correction
        current_step: Current step number
        max_steps: Maximum steps
        num_data_qubits: Total data qubits (both panels)
        num_checks: Number of checks

    Returns:
        node_features: [num_nodes, 10] tensor
    """
    num_nodes = num_data_qubits + 2 * num_checks
    feature_dim = 10

    features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

    # Data qubit features
    features[:num_data_qubits, 0] = 1.0  # is_data
    features[:num_data_qubits, 3] = correction
    features[:num_data_qubits, 4] = current_step / max_steps

    # X-check features
    x_check_start = num_data_qubits
    x_check_end = num_data_qubits + num_checks
    features[x_check_start:x_check_end, 1] = 1.0  # is_X_check
    features[x_check_start:x_check_end, 4] = current_step / max_steps
    features[x_check_start:x_check_end, 5] = syndrome_X
    features[x_check_start:x_check_end, 6] = np.sum(syndrome_X) / num_checks

    # Z-check features
    z_check_start = num_data_qubits + num_checks
    z_check_end = num_nodes
    features[z_check_start:z_check_end, 2] = 1.0  # is_Z_check
    features[z_check_start:z_check_end, 4] = current_step / max_steps
    features[z_check_start:z_check_end, 7] = syndrome_Z
    features[z_check_start:z_check_end, 8] = np.sum(syndrome_Z) / num_checks

    return torch.tensor(features, dtype=torch.float32)
