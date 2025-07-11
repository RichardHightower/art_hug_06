"""Unit tests for Chapter 06 examples."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_device
from named_entity_recognition import run_named_entity_recognition_examples

def test_device_detection():
    """Test that device detection works."""
    device = get_device()
    assert device in ["cpu", "cuda", "mps"]
    
def test_named_entity_recognition_runs():
    """Test that named_entity_recognition examples run without errors."""
    # This is a basic smoke test
    try:
        run_named_entity_recognition_examples()
    except Exception as e:
        pytest.fail(f"named_entity_recognition examples failed: {e}")
        
def test_imports():
    """Test that all required modules can be imported."""
    import transformers
    import torch
    import numpy
    import pandas
    
    assert transformers.__version__
    assert torch.__version__
