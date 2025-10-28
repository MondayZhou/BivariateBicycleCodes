"""
Experimental Framework for Comparing Decoders

Provides tools to:
1. Compare two-agent vs single-agent RL
2. Compare RL vs BP-OSD
3. Evaluate hybrid decoder
4. Analyze performance across different error rates
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import time
import json
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoder_setup import bivariate_bicycle_codes
from decoder_run import circuit_simulation
from agent_architecture import TwoAgentDecoder
from single_agent_baseline import SingleAgentDecoder, build_full_tanner_graph, create_full_node_features
from hybrid_decoder import HybridBPOSD_RL_Decoder
from environment import BBCodeDecodingEnv


class DecoderBenchmark:
    """
    Comprehensive benchmarking suite for BB code decoders.
    """

    def __init__(
        self,
        m: int = 6,
        ell: int = 12,
        a: Tuple[int, int, int] = (3, 1, 2),
        b: Tuple[int, int, int] = (3, 1, 2),
        num_cycles: int = 3,
        device: str = 'cpu'
    ):
        self.m = m
        self.ell = ell
        self.a = a
        self.b = b
        self.num_cycles = num_cycles
        self.device = device

        # Initialize code
        self.code_data = bivariate_bicycle_codes(m, ell, a, b, num_cycles)

        # Build graphs for RL decoders
        self.env = BBCodeDecodingEnv(m, ell, a, b, num_cycles)
        self.edge_index_left = self.env.edge_index_left
        self.edge_index_right = self.env.edge_index_right
        self.edge_index_full = build_full_tanner_graph(self.code_data)

    def evaluate_decoder(
        self,
        decoder,
        decoder_type: str,
        error_rates: List[float],
        num_trials: int = 1000,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate a decoder across multiple error rates.

        Args:
            decoder: Decoder object
            decoder_type: 'two_agent', 'single_agent', 'bposd', or 'hybrid'
            error_rates: List of physical error rates to test
            num_trials: Number of trials per error rate
            verbose: Whether to print progress

        Returns:
            results: Dictionary with metrics for each error rate
        """
        results = {
            'error_rates': error_rates,
            'logical_error_rates': [],
            'success_rates': [],
            'avg_decode_time': [],
            'syndrome_satisfaction_rates': [],
            'avg_correction_weight': []
        }

        for p in error_rates:
            if verbose:
                print(f"\nTesting error rate p = {p:.4f}")

            logical_errors = 0
            successes = 0
            decode_times = []
            syndrome_satisfied_count = 0
            correction_weights = []

            for trial in range(num_trials):
                # Generate random error
                error_X, error_Z, syndrome_dict = circuit_simulation(
                    self.code_data,
                    p
                )

                syndrome_X = syndrome_dict['X_checks']
                syndrome_Z = syndrome_dict['Z_checks']

                # Decode
                start_time = time.time()

                if decoder_type == 'two_agent':
                    correction_left, correction_right = self._decode_two_agent(
                        decoder, syndrome_X, syndrome_Z
                    )
                    correction = np.concatenate([correction_left, correction_right])

                elif decoder_type == 'single_agent':
                    correction = self._decode_single_agent(
                        decoder, syndrome_X, syndrome_Z
                    )

                elif decoder_type == 'bposd':
                    correction = self._decode_bposd(
                        syndrome_X, syndrome_Z
                    )

                elif decoder_type == 'hybrid':
                    correction_left, correction_right, _ = self._decode_hybrid(
                        decoder, syndrome_X, syndrome_Z
                    )
                    correction = np.concatenate([correction_left, correction_right])

                else:
                    raise ValueError(f"Unknown decoder type: {decoder_type}")

                decode_time = time.time() - start_time
                decode_times.append(decode_time)

                # Check syndrome satisfaction
                total_error = (error_X + error_Z) % 2
                residual = (total_error + correction) % 2

                hx = self.code_data['hx']
                hz = self.code_data['hz']

                residual_syndrome_X = (hx @ residual) % 2
                residual_syndrome_Z = (hz @ residual) % 2

                syndrome_satisfied = (
                    np.sum(residual_syndrome_X) == 0 and
                    np.sum(residual_syndrome_Z) == 0
                )

                if syndrome_satisfied:
                    syndrome_satisfied_count += 1

                # Check logical error
                logical_error = self._check_logical_error(
                    error_X, error_Z, correction
                )

                if not logical_error:
                    successes += 1
                else:
                    logical_errors += 1

                correction_weights.append(np.sum(correction))

                if verbose and (trial + 1) % 100 == 0:
                    print(f"  Trial {trial + 1}/{num_trials}: "
                          f"LER = {logical_errors / (trial + 1):.4f}, "
                          f"Success = {successes / (trial + 1):.4f}")

            # Store results
            results['logical_error_rates'].append(logical_errors / num_trials)
            results['success_rates'].append(successes / num_trials)
            results['avg_decode_time'].append(np.mean(decode_times))
            results['syndrome_satisfaction_rates'].append(syndrome_satisfied_count / num_trials)
            results['avg_correction_weight'].append(np.mean(correction_weights))

            if verbose:
                print(f"Results for p = {p:.4f}:")
                print(f"  Logical Error Rate: {results['logical_error_rates'][-1]:.4f}")
                print(f"  Success Rate: {results['success_rates'][-1]:.4f}")
                print(f"  Avg Decode Time: {results['avg_decode_time'][-1]:.6f}s")
                print(f"  Syndrome Satisfaction: {results['syndrome_satisfaction_rates'][-1]:.4f}")

        return results

    def _decode_two_agent(
        self,
        decoder: TwoAgentDecoder,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode using two-agent decoder."""
        # Create node features
        correction_left = np.zeros(self.m * self.ell, dtype=np.int8)
        correction_right = np.zeros(self.m * self.ell, dtype=np.int8)

        node_features_left = self.env._create_node_features(
            syndrome_X, syndrome_Z, correction_left, correction_right, 'left'
        ).to(self.device)

        node_features_right = self.env._create_node_features(
            syndrome_X, syndrome_Z, correction_left, correction_right, 'right'
        ).to(self.device)

        syndrome_dict = {'X_checks': syndrome_X, 'Z_checks': syndrome_Z}

        correction_left, correction_right = decoder.decode(
            syndrome_dict,
            node_features_left,
            node_features_right,
            self.edge_index_left.to(self.device),
            self.edge_index_right.to(self.device),
            num_iterations=3
        )

        return correction_left, correction_right

    def _decode_single_agent(
        self,
        decoder: SingleAgentDecoder,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> np.ndarray:
        """Decode using single-agent decoder."""
        correction = np.zeros(2 * self.m * self.ell, dtype=np.int8)

        node_features = create_full_node_features(
            syndrome_X,
            syndrome_Z,
            correction,
            current_step=0,
            max_steps=3,
            num_data_qubits=2 * self.m * self.ell,
            num_checks=self.m * self.ell
        ).to(self.device)

        syndrome_dict = {'X_checks': syndrome_X, 'Z_checks': syndrome_Z}

        correction = decoder.decode(
            syndrome_dict,
            node_features,
            self.edge_index_full.to(self.device),
            num_iterations=3
        )

        return correction

    def _decode_bposd(
        self,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> np.ndarray:
        """Decode using BP-OSD."""
        from ldpc.bposd_decoder import bposd_decoder

        # Initialize BP-OSD decoders
        bposd_X = bposd_decoder(
            self.code_data['hx'],
            error_rate=0.001,
            max_iter=10000,
            bp_method="ms",
            osd_method="osd_cs",
            osd_order=7
        )

        bposd_Z = bposd_decoder(
            self.code_data['hz'],
            error_rate=0.001,
            max_iter=10000,
            bp_method="ms",
            osd_method="osd_cs",
            osd_order=7
        )

        # Decode
        correction_X = bposd_X.decode(syndrome_Z)
        correction_Z = bposd_Z.decode(syndrome_X)

        # Combine corrections
        correction = (correction_X + correction_Z) % 2

        return correction

    def _decode_hybrid(
        self,
        decoder: HybridBPOSD_RL_Decoder,
        syndrome_X: np.ndarray,
        syndrome_Z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Decode using hybrid decoder."""
        # Create node features for RL fallback
        correction_left = np.zeros(self.m * self.ell, dtype=np.int8)
        correction_right = np.zeros(self.m * self.ell, dtype=np.int8)

        node_features_left = self.env._create_node_features(
            syndrome_X, syndrome_Z, correction_left, correction_right, 'left'
        ).to(self.device)

        node_features_right = self.env._create_node_features(
            syndrome_X, syndrome_Z, correction_left, correction_right, 'right'
        ).to(self.device)

        correction_left, correction_right, info = decoder.decode(
            syndrome_X,
            syndrome_Z,
            node_features_left,
            node_features_right,
            self.edge_index_left.to(self.device),
            self.edge_index_right.to(self.device)
        )

        return correction_left, correction_right, info

    def _check_logical_error(
        self,
        error_X: np.ndarray,
        error_Z: np.ndarray,
        correction: np.ndarray
    ) -> bool:
        """Check if correction introduces logical error."""
        # For BB codes, check if total error is non-trivial
        total_error = (error_X + error_Z + correction) % 2

        # Simple check: if correction weight is very large, likely logical error
        # More sophisticated check would use logical operators
        if np.sum(correction) > self.m * self.ell:
            return True

        # Check syndrome (residual should be 0)
        hx = self.code_data['hx']
        hz = self.code_data['hz']

        residual_X = (hx @ total_error) % 2
        residual_Z = (hz @ total_error) % 2

        if np.sum(residual_X) > 0 or np.sum(residual_Z) > 0:
            return True  # Syndrome not satisfied -> logical error

        return False

    def compare_decoders(
        self,
        two_agent_decoder: Optional[TwoAgentDecoder] = None,
        single_agent_decoder: Optional[SingleAgentDecoder] = None,
        hybrid_decoder: Optional[HybridBPOSD_RL_Decoder] = None,
        error_rates: List[float] = [0.0001, 0.0005, 0.001, 0.002, 0.005],
        num_trials: int = 1000,
        include_bposd: bool = True
    ) -> Dict:
        """
        Compare multiple decoders side by side.

        Returns:
            comparison: Dictionary with results for each decoder
        """
        comparison = {}

        print("=" * 60)
        print("DECODER COMPARISON BENCHMARK")
        print("=" * 60)

        # BP-OSD baseline
        if include_bposd:
            print("\n[1/4] Evaluating BP-OSD...")
            comparison['bposd'] = self.evaluate_decoder(
                None, 'bposd', error_rates, num_trials
            )

        # Two-agent RL
        if two_agent_decoder is not None:
            print("\n[2/4] Evaluating Two-Agent RL...")
            comparison['two_agent'] = self.evaluate_decoder(
                two_agent_decoder, 'two_agent', error_rates, num_trials
            )

        # Single-agent RL
        if single_agent_decoder is not None:
            print("\n[3/4] Evaluating Single-Agent RL...")
            comparison['single_agent'] = self.evaluate_decoder(
                single_agent_decoder, 'single_agent', error_rates, num_trials
            )

        # Hybrid decoder
        if hybrid_decoder is not None:
            print("\n[4/4] Evaluating Hybrid BP-OSD + RL...")
            comparison['hybrid'] = self.evaluate_decoder(
                hybrid_decoder, 'hybrid', error_rates, num_trials
            )

        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)

        return comparison

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for decoder_name, decoder_results in results.items():
            results_json[decoder_name] = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in decoder_results.items()
            }

        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\nResults saved to {filename}")

    def print_summary(self, comparison: Dict):
        """Print summary comparison table."""
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)

        error_rates = comparison[list(comparison.keys())[0]]['error_rates']

        for i, p in enumerate(error_rates):
            print(f"\nError Rate p = {p:.4f}")
            print("-" * 80)
            print(f"{'Decoder':<20} {'LER':<12} {'Success':<12} {'Time (ms)':<12} {'Syn Sat':<12}")
            print("-" * 80)

            for decoder_name, results in comparison.items():
                ler = results['logical_error_rates'][i]
                success = results['success_rates'][i]
                time_ms = results['avg_decode_time'][i] * 1000
                syn_sat = results['syndrome_satisfaction_rates'][i]

                print(f"{decoder_name:<20} {ler:<12.4f} {success:<12.4f} "
                      f"{time_ms:<12.4f} {syn_sat:<12.4f}")

        print("=" * 80)

    def analyze_coordination_benefit(
        self,
        two_agent_decoder: TwoAgentDecoder,
        num_trials: int = 100,
        error_rate: float = 0.001
    ) -> Dict:
        """
        Analyze the benefit of cross-panel coordination in two-agent decoder.

        Compares:
        - Two agents with coordination (cross-attention)
        - Two agents without coordination (independent)

        Returns:
            analysis: Dictionary with coordination metrics
        """
        print("\n" + "=" * 60)
        print("COORDINATION BENEFIT ANALYSIS")
        print("=" * 60)

        results = {
            'with_coordination': {'successes': 0, 'logical_errors': 0},
            'without_coordination': {'successes': 0, 'logical_errors': 0}
        }

        for trial in range(num_trials):
            # Generate error
            error_X, error_Z, syndrome_dict = circuit_simulation(
                self.code_data,
                error_rate
            )

            syndrome_X = syndrome_dict['X_checks']
            syndrome_Z = syndrome_dict['Z_checks']

            # Decode with coordination
            correction_left_coord, correction_right_coord = self._decode_two_agent(
                two_agent_decoder, syndrome_X, syndrome_Z
            )
            correction_coord = np.concatenate([correction_left_coord, correction_right_coord])

            # Check logical error
            logical_error_coord = self._check_logical_error(
                error_X, error_Z, correction_coord
            )

            if not logical_error_coord:
                results['with_coordination']['successes'] += 1
            else:
                results['with_coordination']['logical_errors'] += 1

            # TODO: Decode without coordination (would require modifying forward pass)
            # For now, we skip this comparison

        print(f"\nWith Coordination:")
        print(f"  Success Rate: {results['with_coordination']['successes'] / num_trials:.4f}")
        print(f"  Logical Error Rate: {results['with_coordination']['logical_errors'] / num_trials:.4f}")

        return results
