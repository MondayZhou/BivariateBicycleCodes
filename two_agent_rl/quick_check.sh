#!/bin/bash
# Quick sanity check script
# Run this before starting long experiments to catch errors early

echo "Running quick sanity check..."
echo ""

/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python quick_test.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ Ready to run experiments!"
    echo "  Use: python run_experiments.py [args]"
else
    echo ""
    echo "✗ Issues detected. Please fix errors before running experiments."
fi

exit $exit_code

