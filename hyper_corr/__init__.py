#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:19:47 2025

@author: Jon Paul Lundquist
"""

# hyper_corr/__init__.py
from importlib.metadata import version, PackageNotFoundError

# Public API (update this list if you add new symbols)
__all__ = [
    "pearsonr",
    "spearmanr", "spearmanr_noties", "spearmanr_ties",
    "kendalltau", "kendalltau_noties", "kendalltau_ties",
    "chatterjeexi", "chatterjeexi_noties", "chatterjeexi_ties",
    "somersd", "somersd_noties", "somersd_ties",
    "__version__",
]

# bind functions directly on the package
from .pearsonr import pearsonr
from .spearmanr import spearmanr, spearmanr_noties, spearmanr_ties
from .kendalltau import kendalltau, kendalltau_noties, kendalltau_ties
from .chatterjeexi import chatterjeexi, chatterjeexi_noties, chatterjeexi_ties
from .somersd import somersd, somersd_noties, somersd_ties

# Package version
try:
    __version__ = version("hyper_corr")
except PackageNotFoundError:
    __version__ = "0.0.0"

