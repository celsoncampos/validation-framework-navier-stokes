#!/usr/bin/env python3
"""
INDEPENDENT VALIDATION FRAMEWORK
================================

This package provides complete validation capabilities for the antisymmetric
decomposition paper on 3D Navier-Stokes equations.

Modules:
- antisymmetric_decomposition: Core theory implementation
- spectral_operators: High-precision differential operators  
- incompressible_fields: Divergence-free field generation
- test_cases: Comprehensive validation test cases

Usage:
    from validation.antisymmetric_decomposition import AntisymmetricDecomposer
    from validation.spectral_operators import SpectralOperators
    from validation.incompressible_fields import IncompressibleFieldGenerator
    from validation.test_cases import ValidationTestCases

Author: Celso Campos
Date: August 18, 2025
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Celso Campos"
__email__ = "celsocampos636@gmail.com"

# Import main classes for convenience
from .antisymmetric_decomposition import AntisymmetricDecomposer
from .spectral_operators import SpectralOperators
from .incompressible_fields import IncompressibleFieldGenerator
from .test_cases import ValidationTestCases

__all__ = [
    'AntisymmetricDecomposer',
    'SpectralOperators', 
    'IncompressibleFieldGenerator',
    'ValidationTestCases'
]