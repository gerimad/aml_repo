"""Tests for src/scripts/run_test.py."""

import pytest

import run_test


def test_unknown_model_kind_is_an_error():
    with pytest.raises(ValueError):
        run_test.scorer_factory({"kind": "magic"}, None, None)
