"""
Сканеры паттернов "Флаг"
"""
from .bullish_flag_scanner import BullishFlagScanner
from .bearish_flag_scanner import BearishFlagScanner
from .combined_scanner import ComplexFlagScanner
from .hybrid_scanner import HybridFlagScanner

__all__ = [
    'BullishFlagScanner',
    'BearishFlagScanner',
    'ComplexFlagScanner',
    'HybridFlagScanner'
]
