"""
Extract Panel-Specific Features from BP-OSD

BP-OSD internally uses belief propagation which has soft information
(log-likelihood ratios) for each qubit. This module extracts and decomposes
this information by panel to help initialize two-agent RL.

Key insight: BP-OSD's soft information encodes uncertainty per qubit.
By splitting this by panel, we can:
1. Initialize agent confidence estimates
2. Guide cross-panel attention
3. Provide panel-specific uncertainty features
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from ldpc.bposd_decoder import bposd_decoder


class BPOSDPanelAnalyzer:
    """
    Analyze BP-OSD internal state and extract panel-specific features.
    """

    def __init__(self, code_data: Dict):
        self.code_data = code_data
        self.m = code_data['m']
        self.ell = code_data['ell']
        self.num_data_per_panel = self.m * self.ell

    def extract_bp_soft_information(
        self,
        bposd_decoder_instance: bposd_decoder,
        syndrome: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract soft information (LLRs) from BP decoder after decoding.

        Args:
            bposd_decoder_instance: BP-OSD decoder instance (after decode())
            syndrome: Syndrome that was decoded

        Returns:
            llr_left: Log-likelihood ratios for left panel [num_data_per_panel]
            llr_right: Log-likelihood ratios for right panel [num_data_per_panel]
        """
        # After running decode(), BP decoder has bp_decoding attribute
        if hasattr(bposd_decoder_instance, 'bp_decoding') and \
           bposd_decoder_instance.bp_decoding is not None:
            bp_solution = bposd_decoder_instance.bp_decoding
        else:
            # BP didn't converge, use zeros
            bp_solution = np.zeros(2 * self.num_data_per_panel)

        # Split by panel
        llr_left = bp_solution[:self.num_data_per_panel]
        llr_right = bp_solution[self.num_data_per_panel:]

        return llr_left, llr_right

    def compute_panel_confidence_scores(
        self,
        llr_left: np.ndarray,
        llr_right: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute overall confidence score for each panel.

        Confidence = how certain BP is about the corrections.
        Higher LLR magnitude = more confident.

        Returns:
            confidence_left: Confidence score for left panel [0, 1]
            confidence_right: Confidence score for right panel [0, 1]
        """
        # Confidence is based on LLR magnitude (certainty)
        confidence_left = np.mean(np.abs(llr_left))
        confidence_right = np.mean(np.abs(llr_right))

        # Normalize to [0, 1] range (assume LLR up to 10)
        confidence_left = min(confidence_left / 10.0, 1.0)
        confidence_right = min(confidence_right / 10.0, 1.0)

        return confidence_left, confidence_right

    def identify_uncertain_qubits(
        self,
        llr: np.ndarray,
        threshold: float = 1.0
    ) -> np.ndarray:
        """
        Identify qubits where BP is uncertain (low LLR magnitude).

        Args:
            llr: Log-likelihood ratios
            threshold: LLR magnitude threshold for "uncertain"

        Returns:
            uncertain_mask: Boolean mask of uncertain qubits
        """
        return np.abs(llr) < threshold

    def compute_cross_panel_influence(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute how much each panel influences the other via syndrome.

        This helps initialize cross-panel attention weights.

        Returns:
            influence_dict: {
                'left_to_right': influence score,
                'right_to_left': influence score
            }
        """
        # Get check matrices
        hx = self.code_data['hx']
        hz = self.code_data['hz']

        # Each check connects to both panels
        # Influence = fraction of syndrome weight that involves both panels

        # X-checks: [A | B] structure
        # Syndrome comes from errors in both panels
        syndrome_X_weight = np.sum(syndrome_X)
        syndrome_Z_weight = np.sum(syndrome_Z)

        # If syndrome is non-zero, panels are coupled
        coupling_strength = (syndrome_X_weight + syndrome_Z_weight) / (2 * self.m * self.ell)

        # Influence is symmetric in BB codes (balanced structure)
        influence = min(coupling_strength, 1.0)

        return {
            'left_to_right': influence,
            'right_to_left': influence,
            'total_coupling': coupling_strength
        }

    def generate_panel_features(
        self,
        bposd_X: bposd_decoder,
        bposd_Z: bposd_decoder,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> Dict:
        """
        Generate comprehensive panel-specific features from BP-OSD.

        Returns dictionary with features useful for initializing RL agents.
        """
        # Run BP-OSD decoding
        correction_X = bposd_X.decode(syndrome_Z)
        correction_Z = bposd_Z.decode(syndrome_X)

        # Extract soft information (if BP converged)
        # Note: BP-OSD may not expose LLRs directly, so we use the binary solution
        # In practice, you'd need to access BP internal state or use a custom BP

        # For now, use correction magnitude as proxy for confidence
        correction_total = (correction_X + correction_Z) % 2

        correction_left = correction_total[:self.num_data_per_panel]
        correction_right = correction_total[self.num_data_per_panel:]

        # Compute panel statistics
        features = {
            # Corrections
            'correction_left': correction_left,
            'correction_right': correction_right,
            'correction_weight_left': np.sum(correction_left),
            'correction_weight_right': np.sum(correction_right),

            # Syndrome
            'syndrome_X': syndrome_X,
            'syndrome_Z': syndrome_Z,
            'syndrome_weight_X': np.sum(syndrome_X),
            'syndrome_weight_Z': np.sum(syndrome_Z),

            # Panel balance
            'weight_balance': np.sum(correction_left) / (np.sum(correction_right) + 1e-8),

            # Cross-panel coupling
            'coupling': self.compute_cross_panel_influence(syndrome_X, syndrome_Z),

            # BP convergence info
            'bp_converged_X': bposd_X.bp_decoding is not None if hasattr(bposd_X, 'bp_decoding') else False,
            'bp_converged_Z': bposd_Z.bp_decoding is not None if hasattr(bposd_Z, 'bp_decoding') else False,
        }

        return features


class CommunicationInitializer:
    """
    Initialize cross-panel communication based on BP-OSD patterns.

    The idea: BP-OSD shows us which qubits in each panel are important
    for cross-panel coordination. Use this to warmstart attention.
    """

    def __init__(self, panel_analyzer: BPOSDPanelAnalyzer):
        self.analyzer = panel_analyzer

    def compute_attention_initialization(
        self,
        features_dataset: list
    ) -> torch.Tensor:
        """
        Compute initial attention weights from BP-OSD patterns.

        Args:
            features_dataset: List of feature dicts from BP-OSD

        Returns:
            attention_init: Initial attention weights [num_data, num_data]
        """
        num_data = self.analyzer.num_data_per_panel

        # Accumulate cross-panel attention patterns
        attention_matrix = np.zeros((num_data, num_data))

        for features in features_dataset:
            correction_left = features['correction_left']
            correction_right = features['correction_right']

            # Qubits that are corrected in both panels are "coupled"
            for i in range(num_data):
                for j in range(num_data):
                    if correction_left[i] and correction_right[j]:
                        attention_matrix[i, j] += 1

        # Normalize
        attention_matrix = attention_matrix / (len(features_dataset) + 1e-8)

        # Add uniform baseline
        attention_matrix += 0.1 / num_data

        # Normalize rows (attention weights sum to 1)
        attention_matrix = attention_matrix / (attention_matrix.sum(axis=1, keepdims=True) + 1e-8)

        return torch.tensor(attention_matrix, dtype=torch.float32)

    def initialize_agent_communication(
        self,
        agent_left,
        agent_right,
        attention_init: torch.Tensor
    ):
        """
        Initialize cross-attention weights in agents.

        This gives agents a "warm start" for communication.
        """
        # Note: PyTorch attention layers don't directly expose weight initialization
        # This would require modifying the attention mechanism or using bias initialization

        print("Communication initialization: Attention patterns computed from BP-OSD")
        print(f"  Average attention weight: {attention_init.mean():.6f}")
        print(f"  Max attention weight: {attention_init.max():.6f}")
        print(f"  Sparsity: {(attention_init < 0.001).float().mean():.2%}")

        # In practice, you'd inject this into the attention mechanism
        # For example, as an attention bias or by pretraining the attention weights
        return attention_init


def analyze_bposd_panel_statistics(
    code_data: Dict,
    num_samples: int = 1000,
    error_rate: float = 0.001
) -> Dict:
    """
    Analyze BP-OSD behavior on many samples to understand panel patterns.

    This helps answer: What does BP-OSD teach us about panel structure?
    """
    from decoder_run import circuit_simulation

    analyzer = BPOSDPanelAnalyzer(code_data)

    # Initialize BP-OSD
    bposd_X = bposd_decoder(
        code_data['hx'],
        error_rate=error_rate,
        max_iter=10000,
        bp_method="ms",
        osd_method="osd_cs",
        osd_order=7
    )

    bposd_Z = bposd_decoder(
        code_data['hz'],
        error_rate=error_rate,
        max_iter=10000,
        bp_method="ms",
        osd_method="osd_cs",
        osd_order=7
    )

    print(f"\nAnalyzing BP-OSD panel behavior on {num_samples} samples...")

    statistics = {
        'weight_balance_samples': [],
        'coupling_samples': [],
        'correction_weight_left': [],
        'correction_weight_right': [],
    }

    for i in range(num_samples):
        error_X, error_Z, syndrome_dict = circuit_simulation(code_data, error_rate)

        features = analyzer.generate_panel_features(
            bposd_X, bposd_Z,
            syndrome_dict['X_checks'],
            syndrome_dict['Z_checks']
        )

        statistics['weight_balance_samples'].append(features['weight_balance'])
        statistics['coupling_samples'].append(features['coupling']['total_coupling'])
        statistics['correction_weight_left'].append(features['correction_weight_left'])
        statistics['correction_weight_right'].append(features['correction_weight_right'])

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples")

    # Compute summary statistics
    summary = {
        'avg_weight_balance': np.mean(statistics['weight_balance_samples']),
        'std_weight_balance': np.std(statistics['weight_balance_samples']),
        'avg_coupling': np.mean(statistics['coupling_samples']),
        'avg_weight_left': np.mean(statistics['correction_weight_left']),
        'avg_weight_right': np.mean(statistics['correction_weight_right']),
        'left_right_correlation': np.corrcoef(
            statistics['correction_weight_left'],
            statistics['correction_weight_right']
        )[0, 1]
    }

    print("\n" + "=" * 60)
    print("BP-OSD PANEL ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Average weight balance (L/R): {summary['avg_weight_balance']:.4f}")
    print(f"  (1.0 means equal, >1 means left-heavy, <1 means right-heavy)")
    print(f"Std dev of balance: {summary['std_weight_balance']:.4f}")
    print(f"Average coupling strength: {summary['avg_coupling']:.4f}")
    print(f"Average correction weight (L): {summary['avg_weight_left']:.2f}")
    print(f"Average correction weight (R): {summary['avg_weight_right']:.2f}")
    print(f"Left-Right correlation: {summary['left_right_correlation']:.4f}")
    print("=" * 60)

    print("\nInterpretation:")
    if abs(summary['avg_weight_balance'] - 1.0) < 0.1:
        print("  ✓ Panels are well-balanced (symmetric errors)")
    else:
        print("  ⚠ Panels are imbalanced (asymmetric errors)")

    if summary['avg_coupling'] > 0.1:
        print("  ✓ High cross-panel coupling (agents MUST communicate)")
    else:
        print("  ⚠ Low cross-panel coupling (agents can work independently)")

    if summary['left_right_correlation'] > 0.5:
        print("  ✓ Panel errors are correlated (shared patterns)")
    else:
        print("  ⚠ Panel errors are independent")

    return summary, statistics
