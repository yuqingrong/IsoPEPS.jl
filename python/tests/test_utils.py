"""Tests for utility functions."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tfim.utils import (
    save_results_json,
    load_results_json,
    format_energy_error,
    format_correlation_length,
)


class TestJSONIO:
    """Tests for JSON I/O functions."""

    def test_save_and_load_dict(self):
        """Save and load dictionary results."""
        results = {
            'g': 1.0,
            'energy': -1.2732,
            'xi': np.inf,
        }

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_results_json(results, f.name)
            loaded = load_results_json(f.name)

        assert loaded['g'] == 1.0
        assert np.isclose(loaded['energy'], -1.2732)
        assert loaded['xi'] is None  # inf converted to None

    def test_save_and_load_list(self):
        """Save and load list of results."""
        results = [
            {'g': 0.5, 'energy': -1.0635},
            {'g': 1.0, 'energy': -1.2732},
        ]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_results_json(results, f.name)
            loaded = load_results_json(f.name)

        assert 'results' in loaded
        assert len(loaded['results']) == 2
        assert loaded['results'][0]['g'] == 0.5

    def test_save_numpy_types(self):
        """Numpy types are converted to Python types."""
        results = {
            'g': np.float64(1.0),
            'energy': np.float32(-1.2732),
            'n': np.int64(10),
            'array': np.array([1, 2, 3]),
        }

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_results_json(results, f.name)
            loaded = load_results_json(f.name)

        assert isinstance(loaded['g'], float)
        assert isinstance(loaded['n'], float)
        assert loaded['array'] == [1, 2, 3]

    def test_save_handles_nan_and_inf(self):
        """NaN and inf are converted to None."""
        results = {
            'finite': 1.0,
            'nan': np.nan,
            'inf': np.inf,
            'neg_inf': -np.inf,
        }

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_results_json(results, f.name)
            loaded = load_results_json(f.name)

        assert loaded['finite'] == 1.0
        assert loaded['nan'] is None
        assert loaded['inf'] is None
        assert loaded['neg_inf'] is None

    def test_includes_description(self):
        """Output includes description."""
        results = {'data': [1, 2, 3]}

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_results_json(results, f.name, description="Test results")
            loaded = load_results_json(f.name)

        assert loaded['description'] == "Test results"
        assert 'hamiltonian' in loaded


class TestFormatFunctions:
    """Tests for formatting functions."""

    def test_format_energy_error_small(self):
        """Format very small errors."""
        result = format_energy_error(-1.27323954, -1.27323954)
        assert "< 1e-14" in result or "0" in result

    def test_format_energy_error_normal(self):
        """Format normal errors."""
        result = format_energy_error(-1.273, -1.27323954)
        assert "e" in result.lower()  # Scientific notation

    def test_format_correlation_length_finite(self):
        """Format finite correlation length."""
        assert format_correlation_length(1.44) == "1.44"
        assert format_correlation_length(0.5) == "0.50"

    def test_format_correlation_length_infinite(self):
        """Format infinite correlation length."""
        assert format_correlation_length(np.inf) == "∞"

    def test_format_correlation_length_none(self):
        """Format None correlation length."""
        assert format_correlation_length(None) == "N/A"


class TestNestedConversion:
    """Test nested data structure conversion."""

    def test_nested_dict(self):
        """Convert nested dictionaries."""
        results = {
            'level1': {
                'level2': {
                    'value': np.float64(1.0),
                    'inf': np.inf,
                }
            }
        }

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_results_json(results, f.name)
            loaded = load_results_json(f.name)

        assert loaded['level1']['level2']['value'] == 1.0
        assert loaded['level1']['level2']['inf'] is None

    def test_list_of_dicts(self):
        """Convert list of dictionaries."""
        results = [
            {'g': np.float64(0.5), 'E': np.nan},
            {'g': np.float64(1.0), 'E': np.float64(-1.27)},
        ]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            save_results_json(results, f.name)
            loaded = load_results_json(f.name)

        assert loaded['results'][0]['g'] == 0.5
        assert loaded['results'][0]['E'] is None
        assert loaded['results'][1]['E'] == -1.27
