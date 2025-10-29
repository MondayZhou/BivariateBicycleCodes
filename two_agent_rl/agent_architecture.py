"""
Two-Agent RL Decoder Architecture for Bivariate Bicycle Codes

This module implements GNN-based agents that exploit the two-panel structure
of BB codes, where each panel has three connections per check qubit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, Tuple, Optional
import numpy as np


class PanelAgentGNN(nn.Module):
    """
    Graph Neural Network agent for decoding one panel of BB code.

    Architecture:
    - Node embeddings for qubits and checks
    - Multiple GAT layers for local reasoning along Tanner graph
    - Cross-panel attention for coordination between agents
    - Policy head for action selection
    - Value head for critic (used in PPO training)
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

        # GNN layers for local message passing on Tanner graph
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

        # Cross-panel attention for agent coordination
        self.cross_panel_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

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
        batch: Optional[torch.Tensor] = None,
        partner_encoding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the agent network.

        Args:
            node_features: [num_nodes, feature_dim] - features for data qubits and checks
            edge_index: [2, num_edges] - Tanner graph connectivity
            batch: [num_nodes] - batch assignment for each node (for batched processing)
            partner_encoding: [num_partner_nodes, hidden_dim] - encoded state from partner agent

        Returns:
            logits: [num_data_qubits, 2] - action logits for each data qubit
            value: [1] - state value estimate
            encoding: [num_nodes, hidden_dim] - node embeddings to share with partner
        """
        # Embed node features
        x = self.node_embedding(node_features)  # [num_nodes, hidden_dim]

        # Local reasoning via GNN on Tanner graph
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            x_new = gnn_layer(x, edge_index)
            x_new = F.relu(x_new)
            x = layer_norm(x + x_new)  # Residual connection

        # Cross-panel coordination via attention
        if partner_encoding is not None:
            # Reshape for attention: [batch_size=1, seq_len, hidden_dim]
            x_q = x.unsqueeze(0) if x.dim() == 2 else x
            partner_kv = partner_encoding.unsqueeze(0) if partner_encoding.dim() == 2 else partner_encoding

            x_attended, _ = self.cross_panel_attention(x_q, partner_kv, partner_kv)
            x_attended = x_attended.squeeze(0) if x.dim() == 2 else x_attended
            x = self.cross_attn_norm(x + x_attended)  # Residual connection

        # Extract data qubit embeddings (first num_data_qubits nodes)
        data_qubit_embeddings = x[:self.num_data_qubits]

        # Compute policy logits for data qubits
        logits = self.policy_net(data_qubit_embeddings)  # [num_data_qubits, 2]

        # Compute value estimate (global average pooling over all nodes)
        if batch is not None:
            pooled = global_mean_pool(x, batch)
        else:
            pooled = x.mean(dim=0, keepdim=True)
        value = self.value_net(pooled)  # [1, 1] or [batch_size, 1]

        return logits, value, x

    def get_action_and_log_prob(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        partner_encoding: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and compute log probability.

        Args:
            node_features: Node features
            edge_index: Graph connectivity
            partner_encoding: Partner agent's encoding
            deterministic: If True, take argmax instead of sampling

        Returns:
            actions: [num_data_qubits] - binary actions
            log_probs: [num_data_qubits] - log probabilities of actions
            value: [1] - state value estimate
        """
        logits, value, encoding = self.forward(node_features, edge_index, partner_encoding=partner_encoding)

        # Create categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)

        return actions, log_probs, value


class TwoAgentDecoder(nn.Module):
    """
    Coordinator for two panel agents that jointly decode BB codes.

    The two agents operate on left and right data panels respectively,
    coordinating through shared syndrome information and cross-attention.
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
        self.num_data_qubits_per_panel = m * ell
        self.num_checks = m * ell  # For both X and Z checks

        # Initialize two agents for left and right panels
        self.agent_left = PanelAgentGNN(
            num_data_qubits=self.num_data_qubits_per_panel,
            num_checks=self.num_checks * 2,  # X and Z checks
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )

        self.agent_right = PanelAgentGNN(
            num_data_qubits=self.num_data_qubits_per_panel,
            num_checks=self.num_checks * 2,
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )

    def forward(
        self,
        state_left: Dict[str, torch.Tensor],
        state_right: Dict[str, torch.Tensor],
        num_iterations: int = 3,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Multi-round decoding with iterative communication between agents.

        Args:
            state_left: State dict for left panel with 'node_features', 'edge_index'
            state_right: State dict for right panel
            num_iterations: Number of iterative refinement rounds
            deterministic: Whether to use deterministic policy

        Returns:
            correction_left: [num_data_qubits] - correction for left panel
            correction_right: [num_data_qubits] - correction for right panel
            info: Dict with intermediate information (log_probs, values, etc.)
        """
        # Initialize corrections
        correction_left = torch.zeros(self.num_data_qubits_per_panel, dtype=torch.long)
        correction_right = torch.zeros(self.num_data_qubits_per_panel, dtype=torch.long)

        # Track information for training
        log_probs_left_list = []
        log_probs_right_list = []
        values_left_list = []
        values_right_list = []
        actions_left_list = []
        actions_right_list = []

        encoding_left = None
        encoding_right = None

        for iteration in range(num_iterations):
            # Agent left acts
            actions_left, log_probs_left, value_left = self.agent_left.get_action_and_log_prob(
                state_left['node_features'],
                state_left['edge_index'],
                partner_encoding=encoding_right,
                deterministic=deterministic
            )

            # Get encoding from agent left
            _, _, encoding_left = self.agent_left(
                state_left['node_features'],
                state_left['edge_index'],
                partner_encoding=encoding_right
            )

            # Agent right acts with knowledge of agent left's encoding
            actions_right, log_probs_right, value_right = self.agent_right.get_action_and_log_prob(
                state_right['node_features'],
                state_right['edge_index'],
                partner_encoding=encoding_left,
                deterministic=deterministic
            )

            # Get encoding from agent right
            _, _, encoding_right = self.agent_right(
                state_right['node_features'],
                state_right['edge_index'],
                partner_encoding=encoding_left
            )

            # Update corrections (XOR for Pauli operations)
            correction_left ^= actions_left
            correction_right ^= actions_right

            # Store for training
            log_probs_left_list.append(log_probs_left)
            log_probs_right_list.append(log_probs_right)
            values_left_list.append(value_left)
            values_right_list.append(value_right)
            actions_left_list.append(actions_left)
            actions_right_list.append(actions_right)

        info = {
            'log_probs_left': torch.stack(log_probs_left_list),
            'log_probs_right': torch.stack(log_probs_right_list),
            'values_left': torch.stack(values_left_list),
            'values_right': torch.stack(values_right_list),
            'actions_left': torch.stack(actions_left_list),
            'actions_right': torch.stack(actions_right_list),
        }

        return correction_left, correction_right, info

    def decode(
        self,
        syndrome_dict: Dict[str, np.ndarray],
        node_features_left: torch.Tensor,
        node_features_right: torch.Tensor,
        edge_index_left: torch.Tensor,
        edge_index_right: torch.Tensor,
        num_iterations: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference mode decoding (no gradients, deterministic).

        Args:
            syndrome_dict: Dictionary with 'X_checks' and 'Z_checks'
            node_features_left: Node features for left panel
            node_features_right: Node features for right panel
            edge_index_left: Tanner graph for left panel
            edge_index_right: Tanner graph for right panel
            num_iterations: Number of refinement iterations

        Returns:
            correction_left: Numpy array of corrections for left panel
            correction_right: Numpy array of corrections for right panel
        """
        self.eval()
        with torch.no_grad():
            state_left = {
                'node_features': node_features_left,
                'edge_index': edge_index_left
            }
            state_right = {
                'node_features': node_features_right,
                'edge_index': edge_index_right
            }

            correction_left, correction_right, _ = self.forward(
                state_left,
                state_right,
                num_iterations=num_iterations,
                deterministic=True
            )

        return correction_left.cpu().numpy(), correction_right.cpu().numpy()
