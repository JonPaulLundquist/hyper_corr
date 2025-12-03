"""Smoke tests for the public import surface of hyper_corr."""

import importlib


def test_public_imports_available():
    module = importlib.import_module("hyper_corr")

    for name in [
        "pearsonr",
        "spearmanr",
        "spearmanr_noties",
        "spearmanr_ties",
        "kendalltau",
        "kendalltau_noties",
        "kendalltau_ties",
        "chatterjeexi",
        "chatterjeexi_noties",
        "chatterjeexi_ties",
        "somersd",
        "somersd_noties",
        "somersd_ties",
    ]:
        assert hasattr(module, name), f"hyper_corr missing {name}"


def test_version_importable():
    module = importlib.import_module("hyper_corr")
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)
