#!/bin/bash
# Quick test script for two-agent RL decoder
# Runs a fast demo to verify installation and functionality

echo "=================================="
echo "Two-Agent RL Decoder - Quick Test"
echo "=================================="
echo ""
echo "This will:"
echo "  1. Check dependencies"
echo "  2. Train for 5,000 steps (~2 min)"
echo "  3. Evaluate on 2 error rates (~2 min)"
echo "  4. Generate comparison plots"
echo ""
echo "Total time: ~5 minutes"
echo ""
read -p "Press Enter to continue..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found!"
    exit 1
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || { echo "✗ PyTorch not installed"; exit 1; }
python -c "import torch_geometric; print('✓ PyTorch Geometric')" || { echo "✗ PyTorch Geometric not installed"; exit 1; }
python -c "import numpy; print('✓ NumPy')" || { echo "✗ NumPy not installed"; exit 1; }
python -c "import scipy; print('✓ SciPy')" || { echo "✗ SciPy not installed"; exit 1; }
python -c "import matplotlib; print('✓ Matplotlib')" || { echo "✗ Matplotlib not installed"; exit 1; }

echo ""
echo "✓ All dependencies found!"
echo ""

# Run quick experiment
echo "Starting quick experiment..."
echo ""

python run_experiments.py --mode quick

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ Test completed successfully!"
    echo "=================================="
    echo ""
    echo "Check the results/ directory for outputs"
    echo ""
    echo "Next steps:"
    echo "  - Run full experiment: python run_experiments.py --mode full"
    echo "  - See QUICKSTART.md for more options"
else
    echo ""
    echo "=================================="
    echo "✗ Test failed"
    echo "=================================="
    echo ""
    echo "Please check the error messages above"
    echo "See QUICKSTART.md for troubleshooting"
fi
