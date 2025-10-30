#!/usr/bin/env python3
"""
Quick sanity check to verify the code can run without errors.
Tests key integration points without running full training.
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_code_setup():
    """Test 1: BB code initialization"""
    print("Test 1: Initializing BB code... ", end="")
    from decoder_setup import bivariate_bicycle_codes
    
    code_data = bivariate_bicycle_codes(
        m=6, ell=12, 
        a=(3, 1, 2), b=(3, 1, 2), 
        num_cycles=3, 
        error_rate=0.001
    )
    
    # Check required keys exist
    required_keys = ['hx', 'hz', 'logicOp_X', 'logicOp_Z', 
                     'X_checks', 'Z_checks', 'nbs', 'm', 'ell']
    for key in ['hx', 'hz', 'nbs', 'm', 'ell']:
        assert key in code_data, f"Missing key: {key}"
    
    print("✓ PASSED")
    return code_data

def test_syndrome_generation(code_data):
    """Test 2: Syndrome generation"""
    print("Test 2: Generating syndromes... ", end="")
    from decoder_run import circuit_simulation
    
    error_X, error_Z, syndrome_dict = circuit_simulation(code_data, 0.001)
    
    # Check syndrome dictionary structure
    assert 'X_checks' in syndrome_dict, "Missing X_checks"
    assert 'Z_checks' in syndrome_dict, "Missing Z_checks"
    assert 'syndrome_X' in syndrome_dict, "Missing syndrome_X"
    assert 'syndrome_Z' in syndrome_dict, "Missing syndrome_Z"
    
    # Check syndrome shapes
    m, ell = code_data['m'], code_data['ell']
    n_checks = m * ell
    assert len(syndrome_dict['X_checks']) == n_checks, f"X_checks shape wrong: {len(syndrome_dict['X_checks'])} != {n_checks}"
    assert len(syndrome_dict['Z_checks']) == n_checks, f"Z_checks shape wrong: {len(syndrome_dict['Z_checks'])} != {n_checks}"
    
    print("✓ PASSED")
    return error_X, error_Z, syndrome_dict

def test_environment_init(code_data):
    """Test 3: Environment initialization"""
    print("Test 3: Initializing environment... ", end="")
    from environment import BBCodeDecodingEnv
    
    env = BBCodeDecodingEnv(
        m=6, ell=12,
        a=(3, 1, 2), b=(3, 1, 2),
        num_cycles=3,
        error_rate=0.001,
        max_steps=5
    )
    
    # Test reset
    state_left, state_right = env.reset()
    
    # Check state structure
    assert 'node_features' in state_left
    assert 'edge_index' in state_left
    assert 'syndrome_X' in state_left
    assert 'syndrome_Z' in state_left
    
    print("✓ PASSED")
    return env

def test_environment_step(env):
    """Test 4: Environment step"""
    print("Test 4: Testing environment step... ", end="")
    
    state_left, state_right = env.reset()
    
    # Random actions
    action_left = np.random.randint(0, 2, size=env.num_data_qubits_per_panel, dtype=np.int8)
    action_right = np.random.randint(0, 2, size=env.num_data_qubits_per_panel, dtype=np.int8)
    
    # Take step
    next_state_left, next_state_right, reward_left, reward_right, done, info = env.step(
        action_left, action_right
    )
    
    assert isinstance(reward_left, (int, float))
    assert isinstance(done, bool)
    
    print("✓ PASSED")

def test_bposd_decoder(code_data, syndrome_dict):
    """Test 5: BP-OSD decoder"""
    print("Test 5: Testing BP-OSD decoder... ", end="")
    from ldpc.bposd_decoder import bposd_decoder
    
    syndrome_X = syndrome_dict['X_checks']
    syndrome_Z = syndrome_dict['Z_checks']
    
    # Initialize decoders
    bposd_X = bposd_decoder(
        code_data['hz'],
        error_rate=0.001,
        max_iter=100,
        bp_method="ms",
        osd_method="osd0",
        osd_order=0
    )
    
    bposd_Z = bposd_decoder(
        code_data['hx'],
        error_rate=0.001,
        max_iter=100,
        bp_method="ms",
        osd_method="osd0",
        osd_order=0
    )
    
    # Decode
    correction_X = bposd_X.decode(syndrome_Z)
    correction_Z = bposd_Z.decode(syndrome_X)
    
    # Check shapes
    m, ell = code_data['m'], code_data['ell']
    expected_length = 2 * m * ell
    assert len(correction_X) == expected_length, f"correction_X length wrong: {len(correction_X)} != {expected_length}"
    assert len(correction_Z) == expected_length, f"correction_Z length wrong: {len(correction_Z)} != {expected_length}"
    
    print("✓ PASSED")

def test_agent_architecture():
    """Test 6: Agent architecture"""
    print("Test 6: Testing agent architecture... ", end="")
    from agent_architecture import TwoAgentDecoder
    
    decoder = TwoAgentDecoder(
        m=6, ell=12,
        node_feature_dim=10,
        hidden_dim=64,
        num_gnn_layers=2
    )
    
    # Create dummy inputs
    batch_size = 1
    num_nodes = 72 * 3  # data + X_checks + Z_checks
    node_features = torch.randn(num_nodes, 10)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    
    state_left = {
        'node_features': node_features,
        'edge_index': edge_index
    }
    state_right = {
        'node_features': node_features,
        'edge_index': edge_index
    }
    
    # Forward pass
    decoder.eval()
    with torch.no_grad():
        correction_left, correction_right, _ = decoder.forward(
            state_left, state_right, num_iterations=1, deterministic=True
        )
    
    assert correction_left.shape[0] == 72
    assert correction_right.shape[0] == 72
    
    print("✓ PASSED")
    return decoder

def test_hybrid_decoder(code_data, decoder):
    """Test 7: Hybrid decoder"""
    print("Test 7: Testing hybrid decoder... ", end="")
    from hybrid_decoder import HybridBPOSD_RL_Decoder
    
    hybrid = HybridBPOSD_RL_Decoder(
        code_data=code_data,
        rl_decoder=decoder,
        bp_max_iter=100,
        osd_order=0
    )
    
    # Create test syndromes
    m, ell = code_data['m'], code_data['ell']
    syndrome_X = np.random.randint(0, 2, m * ell)
    syndrome_Z = np.random.randint(0, 2, m * ell)
    
    # Create dummy node features
    num_nodes = m * ell * 3
    node_features_left = torch.randn(num_nodes, 10)
    node_features_right = torch.randn(num_nodes, 10)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    
    # Decode (should use BP-OSD by default)
    correction_left, correction_right, info = hybrid.decode(
        syndrome_X, syndrome_Z,
        node_features_left, node_features_right,
        edge_index, edge_index
    )
    
    assert len(correction_left) == m * ell
    assert len(correction_right) == m * ell
    assert 'decoder_used' in info
    
    print(f"✓ PASSED (used: {info['decoder_used']})")

def main():
    print("="*60)
    print("QUICK SANITY CHECK - Testing Key Components")
    print("="*60)
    print()
    
    try:
        # Run tests
        code_data = test_code_setup()
        error_X, error_Z, syndrome_dict = test_syndrome_generation(code_data)
        env = test_environment_init(code_data)
        test_environment_step(env)
        test_bposd_decoder(code_data, syndrome_dict)
        decoder = test_agent_architecture()
        test_hybrid_decoder(code_data, decoder)
        
        print()
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print()
        print("The code is ready to run. You can now execute:")
        print("  python run_experiments.py")
        print()
        return 0
        
    except Exception as e:
        print(f"\n✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

