from pathlib import Path
import pytest
from agate.example import main


def test_example_script(tmp_path):
    """Test example script runs without errors and produces plots"""
    main(tmp_path)
    assert Path(f"{tmp_path}/gas_fractions_for_different_models.pdf").is_file()
    assert Path(f"{tmp_path}/gas_fraction_model_error.pdf").is_file()
