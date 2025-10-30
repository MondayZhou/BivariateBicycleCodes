"""
Hybrid BP-OSD + RL Decoder

Combines classical BP-OSD decoder with two-agent RL for hard cases.

Strategy:
1. First attempt: Use fast BP-OSD decoder
2. If BP-OSD fails or produces high-weight solution: Switch to RL agents
3. RL agents can learn to handle difficult error patterns that BP-OSD struggles with
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc.bposd_decoder import bposd_decoder
from agent_architecture import TwoAgentDecoder


class HybridBPOSD_RL_Decoder:
    """
    Hybrid decoder that combines BP-OSD with two-agent RL.

    The decoder first attempts BP-OSD. If it fails or produces questionable results,
    it falls back to the learned RL agents which may be better at handling
    specific error patterns.
    """

    def __init__(
        self,
        code_data: Dict,
        rl_decoder: TwoAgentDecoder,
        bp_max_iter: int = 10000,
        bp_method: str = "ms",
        osd_method: str = "osd_cs",
        osd_order: int = 7,
        weight_threshold: int = None,  # If BP-OSD solution has weight > threshold, use RL
        confidence_threshold: float = 0.5,  # Confidence score below which we use RL
        use_rl_for_failures: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize hybrid decoder.

        Args:
            code_data: BB code data from decoder_setup.py
            rl_decoder: Trained two-agent RL decoder
            bp_max_iter: Maximum BP iterations
            bp_method: BP method ('ms', 'ps', etc.)
            osd_method: OSD method
            osd_order: OSD order
            weight_threshold: Max acceptable weight for BP-OSD solution
            confidence_threshold: Min confidence for accepting BP-OSD
            use_rl_for_failures: Whether to use RL when BP-OSD fails
            device: Device for RL decoder
        """
        self.code_data = code_data
        self.rl_decoder = rl_decoder
        self.device = device

        # BP-OSD parameters
        self.bp_max_iter = bp_max_iter
        self.bp_method = bp_method
        self.osd_method = osd_method
        self.osd_order = osd_order

        # Hybrid parameters
        m = code_data['m']
        ell = code_data['ell']
        self.weight_threshold = weight_threshold or (m * ell // 4)  # Default: 25% of qubits
        self.confidence_threshold = confidence_threshold
        self.use_rl_for_failures = use_rl_for_failures

        # Initialize BP-OSD decoders
        self._initialize_bposd_decoders()

        # Statistics tracking
        self.stats = {
            'total_decodes': 0,
            'bposd_used': 0,
            'rl_used': 0,
            'bposd_success': 0,
            'rl_success': 0,
            'bposd_failures': 0,
            'rl_after_bposd_failure': 0,
            'high_weight_switches': 0,
            'low_confidence_switches': 0
        }

    def _initialize_bposd_decoders(self):
        """Initialize BP-OSD decoders for X and Z errors."""
        # Use basic parity check matrices hx and hz
        # These work with per-check syndromes (not full syndrome history)
        hx = self.code_data['hx']
        hz = self.code_data['hz']
        
        error_rate = self.code_data.get('error_rate', 0.001)

        # Initialize X-error decoder (uses hz to decode X errors from Z-checks)
        self.bposd_X = bposd_decoder(
            hz,
            error_rate=error_rate,
            max_iter=self.bp_max_iter,
            bp_method=self.bp_method,
            osd_method=self.osd_method,
            osd_order=self.osd_order
        )

        # Initialize Z-error decoder (uses hx to decode Z errors from X-checks)
        self.bposd_Z = bposd_decoder(
            hx,
            error_rate=error_rate,
            max_iter=self.bp_max_iter,
            bp_method=self.bp_method,
            osd_method=self.osd_method,
            osd_order=self.osd_order
        )

    def _bposd_decode(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Attempt decoding with BP-OSD.

        Returns:
            correction_X: X-error correction
            correction_Z: Z-error correction
            info: Dictionary with BP-OSD information
        """
        # Decode X errors (from Z syndrome)
        correction_X = self.bposd_X.decode(syndrome_Z)

        # Decode Z errors (from X syndrome)
        correction_Z = self.bposd_Z.decode(syndrome_X)

        # Extract BP-OSD information
        info = {
            'bp_converged_X': self.bposd_X.bp_decoding is not None,
            'bp_converged_Z': self.bposd_Z.bp_decoding is not None,
            'osd_used_X': True,  # OSD always runs after BP
            'osd_used_Z': True,
            'correction_weight_X': np.sum(correction_X),
            'correction_weight_Z': np.sum(correction_Z),
        }

        return correction_X, correction_Z, info

    def _compute_confidence(
        self,
        correction: np.ndarray,
        syndrome: np.ndarray,
        check_matrix: np.ndarray,
        bp_converged: bool
    ) -> float:
        """
        Compute confidence score for BP-OSD solution.

        Heuristics:
        - BP convergence increases confidence
        - Low correction weight increases confidence
        - Syndrome satisfaction is required
        - Soft information from BP (if available)

        Returns:
            confidence: Score in [0, 1]
        """
        confidence = 0.5  # Base confidence

        # Check syndrome satisfaction
        residual_syndrome = (check_matrix @ correction) % 2
        syndrome_satisfied = np.array_equal(residual_syndrome, syndrome)

        if not syndrome_satisfied:
            return 0.0  # No confidence if syndrome not satisfied

        # BP convergence bonus
        if bp_converged:
            confidence += 0.3

        # Low weight bonus
        weight = np.sum(correction)
        if weight < self.weight_threshold / 2:
            confidence += 0.2
        elif weight > self.weight_threshold:
            confidence -= 0.3

        return np.clip(confidence, 0.0, 1.0)

    def _rl_decode(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray,
        node_features_left: torch.Tensor,
        node_features_right: torch.Tensor,
        edge_index_left: torch.Tensor,
        edge_index_right: torch.Tensor,
        num_iterations: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode using two-agent RL decoder.

        Returns:
            correction_left: Correction for left panel
            correction_right: Correction for right panel
        """
        syndrome_dict = {
            'X_checks': syndrome_X,
            'Z_checks': syndrome_Z
        }

        correction_left, correction_right = self.rl_decoder.decode(
            syndrome_dict,
            node_features_left,
            node_features_right,
            edge_index_left,
            edge_index_right,
            num_iterations=num_iterations
        )

        return correction_left, correction_right

    def decode(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray,
        node_features_left: Optional[torch.Tensor] = None,
        node_features_right: Optional[torch.Tensor] = None,
        edge_index_left: Optional[torch.Tensor] = None,
        edge_index_right: Optional[torch.Tensor] = None,
        force_rl: bool = False,
        syndrome_X_checks: Optional[np.ndarray] = None,
        syndrome_Z_checks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Hybrid decoding: Try BP-OSD first, fall back to RL if needed.

        Args:
            syndrome_X: X-check syndrome (can be full history, but per-check preferred)
            syndrome_Z: Z-check syndrome (can be full history, but per-check preferred)
            node_features_left: Node features for RL (if available)
            node_features_right: Node features for RL (if available)
            edge_index_left: Edge index for RL (if available)
            edge_index_right: Edge index for RL (if available)
            force_rl: Force use of RL decoder
            syndrome_X_checks: Per-check X syndromes (preferred for both BP-OSD and RL)
            syndrome_Z_checks: Per-check Z syndromes (preferred for both BP-OSD and RL)

        Returns:
            correction_left: Correction for left panel
            correction_right: Correction for right panel
            info: Dictionary with decoding information
        """
        self.stats['total_decodes'] += 1

        info = {
            'decoder_used': None,
            'switch_reason': None
        }

        # Use per-check syndromes for both BP-OSD and RL
        # BP-OSD with basic hx/hz matrices expects per-check syndromes
        bposd_syndrome_X = syndrome_X_checks if syndrome_X_checks is not None else syndrome_X
        bposd_syndrome_Z = syndrome_Z_checks if syndrome_Z_checks is not None else syndrome_Z
        rl_syndrome_X = syndrome_X_checks if syndrome_X_checks is not None else syndrome_X
        rl_syndrome_Z = syndrome_Z_checks if syndrome_Z_checks is not None else syndrome_Z
        
        # Option 1: Forced RL usage
        if force_rl:
            correction_left, correction_right = self._rl_decode(
                rl_syndrome_X, rl_syndrome_Z,
                node_features_left, node_features_right,
                edge_index_left, edge_index_right
            )
            info['decoder_used'] = 'RL'
            info['switch_reason'] = 'forced'
            self.stats['rl_used'] += 1
            return correction_left, correction_right, info

        # Option 2: Try BP-OSD first
        correction_X, correction_Z, bposd_info = self._bposd_decode(bposd_syndrome_X, bposd_syndrome_Z)

        # Split corrections into panels
        m_ell = self.code_data['m'] * self.code_data['ell']
        correction_left_bposd = (correction_X[:m_ell] + correction_Z[:m_ell]) % 2
        correction_right_bposd = (correction_X[m_ell:] + correction_Z[m_ell:]) % 2

        # Compute confidence scores
        # correction_X was decoded using hz, so check with hz
        confidence_X = self._compute_confidence(
            correction_X,
            bposd_syndrome_Z,
            self.code_data['hz'],
            bposd_info['bp_converged_X']
        )

        # correction_Z was decoded using hx, so check with hx
        confidence_Z = self._compute_confidence(
            correction_Z,
            bposd_syndrome_X,
            self.code_data['hx'],
            bposd_info['bp_converged_Z']
        )

        confidence = min(confidence_X, confidence_Z)
        total_weight = bposd_info['correction_weight_X'] + bposd_info['correction_weight_Z']

        # Decision: Accept BP-OSD or switch to RL?
        use_rl = False
        switch_reason = None

        # Reason 1: High weight solution
        if total_weight > self.weight_threshold:
            use_rl = True
            switch_reason = 'high_weight'
            self.stats['high_weight_switches'] += 1

        # Reason 2: Low confidence
        elif confidence < self.confidence_threshold:
            use_rl = True
            switch_reason = 'low_confidence'
            self.stats['low_confidence_switches'] += 1

        # Reason 3: BP-OSD failed to satisfy syndrome
        elif confidence == 0.0 and self.use_rl_for_failures:
            use_rl = True
            switch_reason = 'bposd_failure'
            self.stats['bposd_failures'] += 1
            self.stats['rl_after_bposd_failure'] += 1

        # Execute decision
        if use_rl and (node_features_left is not None):
            correction_left, correction_right = self._rl_decode(
                rl_syndrome_X, rl_syndrome_Z,
                node_features_left, node_features_right,
                edge_index_left, edge_index_right
            )
            info['decoder_used'] = 'RL'
            info['switch_reason'] = switch_reason
            info.update(bposd_info)
            info['bposd_confidence'] = confidence
            self.stats['rl_used'] += 1
        else:
            correction_left = correction_left_bposd
            correction_right = correction_right_bposd
            info['decoder_used'] = 'BP-OSD'
            info.update(bposd_info)
            info['confidence'] = confidence
            self.stats['bposd_used'] += 1

        return correction_left, correction_right, info

    def get_statistics(self) -> Dict:
        """Return decoder usage statistics."""
        stats = self.stats.copy()

        if stats['total_decodes'] > 0:
            stats['bposd_usage_rate'] = stats['bposd_used'] / stats['total_decodes']
            stats['rl_usage_rate'] = stats['rl_used'] / stats['total_decodes']

        if stats['bposd_used'] > 0:
            stats['bposd_success_rate'] = stats['bposd_success'] / stats['bposd_used']

        if stats['rl_used'] > 0:
            stats['rl_success_rate'] = stats['rl_success'] / stats['rl_used']

        return stats

    def reset_statistics(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0

    def print_statistics(self):
        """Print decoder usage statistics."""
        stats = self.get_statistics()

        print("\n=== Hybrid Decoder Statistics ===")
        print(f"Total decodes: {stats['total_decodes']}")
        print(f"\nDecoder Usage:")
        print(f"  BP-OSD: {stats['bposd_used']} ({stats.get('bposd_usage_rate', 0):.1%})")
        print(f"  RL:     {stats['rl_used']} ({stats.get('rl_usage_rate', 0):.1%})")
        print(f"\nSwitch Reasons:")
        print(f"  High weight:     {stats['high_weight_switches']}")
        print(f"  Low confidence:  {stats['low_confidence_switches']}")
        print(f"  BP-OSD failures: {stats['bposd_failures']}")
        print(f"\nSuccess Rates:")
        if 'bposd_success_rate' in stats:
            print(f"  BP-OSD: {stats['bposd_success_rate']:.1%}")
        if 'rl_success_rate' in stats:
            print(f"  RL:     {stats['rl_success_rate']:.1%}")
        print("=" * 33)


class AdaptiveHybridDecoder(HybridBPOSD_RL_Decoder):
    """
    Adaptive version of hybrid decoder that learns when to switch.

    Uses a simple learned classifier to predict whether RL will outperform BP-OSD.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simple MLP classifier: syndrome features -> {use_bposd, use_rl}
        self.switch_predictor = torch.nn.Sequential(
            torch.nn.Linear(10, 64),  # Input: syndrome statistics
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),  # Output: [bposd_score, rl_score]
            torch.nn.Softmax(dim=-1)
        ).to(self.device)

        self.switch_predictor_optimizer = torch.optim.Adam(
            self.switch_predictor.parameters(),
            lr=1e-3
        )

        # Training buffer for switch predictor
        self.predictor_buffer = []

    def _extract_syndrome_features(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> torch.Tensor:
        """Extract features from syndrome for switch prediction."""
        features = [
            np.sum(syndrome_X),  # X-syndrome weight
            np.sum(syndrome_Z),  # Z-syndrome weight
            np.max(syndrome_X),  # Max X-syndrome
            np.max(syndrome_Z),  # Max Z-syndrome
            np.mean(syndrome_X),  # Mean X-syndrome
            np.mean(syndrome_Z),  # Mean Z-syndrome
            np.std(syndrome_X),  # Std X-syndrome
            np.std(syndrome_Z),  # Std Z-syndrome
            len(np.nonzero(syndrome_X)[0]) / len(syndrome_X),  # X-density
            len(np.nonzero(syndrome_Z)[0]) / len(syndrome_Z),  # Z-density
        ]
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def predict_switch(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Predict whether to use RL decoder.

        Returns:
            use_rl: Boolean prediction
            rl_score: Confidence in using RL [0, 1]
        """
        features = self._extract_syndrome_features(syndrome_X, syndrome_Z)

        with torch.no_grad():
            scores = self.switch_predictor(features.unsqueeze(0)).squeeze()
            rl_score = scores[1].item()

        use_rl = rl_score > 0.5

        return use_rl, rl_score

    def update_switch_predictor(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray,
        bposd_success: bool,
        rl_success: bool
    ):
        """
        Update switch predictor based on outcome.

        Args:
            syndrome_X: X syndrome
            syndrome_Z: Z syndrome
            bposd_success: Whether BP-OSD succeeded
            rl_success: Whether RL succeeded
        """
        features = self._extract_syndrome_features(syndrome_X, syndrome_Z)

        # Determine target: which decoder was better?
        if rl_success and not bposd_success:
            target = torch.tensor([0.0, 1.0], device=self.device)  # Use RL
        elif bposd_success and not rl_success:
            target = torch.tensor([1.0, 0.0], device=self.device)  # Use BP-OSD
        elif bposd_success and rl_success:
            target = torch.tensor([0.8, 0.2], device=self.device)  # Prefer BP-OSD (faster)
        else:
            return  # Both failed, no clear signal

        # Update predictor
        scores = self.switch_predictor(features.unsqueeze(0)).squeeze()
        loss = torch.nn.functional.mse_loss(scores, target)

        self.switch_predictor_optimizer.zero_grad()
        loss.backward()
        self.switch_predictor_optimizer.step()
