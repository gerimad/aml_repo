"""Tests for the guards of src/scripts/run_test.py (the test period is scored once, from a frozen commit)."""

import subprocess

import pytest

import run_test


def fake_git(answers):
    """A stand-in for run_test.git that answers by the first argument."""

    def git(*args):
        answer = answers[args[0] if args[0] != "rev-parse" else args[1]]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return git


CLEAN = {"status": "?? notes.txt", "HEAD": "abc123", "test-freeze^{commit}": "abc123"}


def test_runs_when_frozen_clean_and_not_yet_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(run_test, "git", fake_git(CLEAN))
    assert run_test.check_frozen(tmp_path / "test") == "abc123"


def test_refuses_uncommitted_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(run_test, "git", fake_git({**CLEAN, "status": " M src/scripts/protocol.py"}))
    with pytest.raises(SystemExit, match="Uncommitted"):
        run_test.check_frozen(tmp_path / "test")


def test_refuses_a_head_that_is_not_the_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(run_test, "git", fake_git({**CLEAN, "HEAD": "def456"}))
    with pytest.raises(SystemExit, match="not the commit tagged"):
        run_test.check_frozen(tmp_path / "test")


def test_refuses_without_the_tag(tmp_path, monkeypatch):
    missing = subprocess.CalledProcessError(128, "git")
    monkeypatch.setattr(run_test, "git", fake_git({**CLEAN, "test-freeze^{commit}": missing}))
    with pytest.raises(SystemExit, match="No tag"):
        run_test.check_frozen(tmp_path / "test")


def test_refuses_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_test, "git", fake_git(CLEAN))
    (tmp_path / "test").mkdir()
    with pytest.raises(SystemExit, match="scored once"):
        run_test.check_frozen(tmp_path / "test")


def test_unknown_model_kind_is_an_error():
    with pytest.raises(ValueError):
        run_test.scorer_factory({"kind": "magic"}, None, None)
