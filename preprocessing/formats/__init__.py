"""
Election data format parsers.
Each parser converts election data to the standard format needed by compare_ks.py
"""

from .base_parser import ElectionData, BaseParser

__all__ = ['ElectionData', 'BaseParser']
