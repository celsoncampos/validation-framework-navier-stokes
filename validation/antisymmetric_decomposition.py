#!/usr/bin/env python3
"""
ANTISYMMETRIC DECOMPOSITION MODULE
=================================

Core implementation of the antisymmetric decomposition theory for 3D Navier-Stokes
equations as described in the paper "A New Approach to Three-Dimensional 
Navier-Stokes Equations via Antisymmetric Decomposition for a Restricted Class 
of Initial Data".

This module implements:
1. Bilateral reflection operator σ(x,y,z) = (x,-y,z)
2. Orthogonal decomposition V = V_antisym ⊕ V_sym
3. Projectors P_a and P_s
4. Fundamental cancellation property verification

Author: Celso Campos
Date: August 18, 2025
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

class AntisymmetricDecomposer:
    """
    Implementation of antisymmetric decomposition for velocity fields
    
    This class handles the core mathematical operations required for the
    antisymmetric decomposition theory including bilateral reflection,
    projection operators, and orthogonality verification.
    """
    
    def __init__(self, spectral_ops):
        """
        Initialize the antisymmetric decomposer
        
        Args:
            spectral_ops: SpectralOperators instance for high-precision calculations
        """
        self.spectral_ops = spectral_ops
        self.logger = logging.getLogger(__name__)
        
    def bilateral_reflection(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply bilateral reflection operator σ(x,y,z) = (x,-y,z)
        
        For periodic domains, this corresponds to:
        - Inverting the y-coordinate index
        - Negating the v-component (y-velocity)
        - Preserving u and w components
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Tuple of transformed velocity components (u_σ, v_σ, w_σ)
        """
        # σ(x,y,z) = (x,-y,z) transformation
        u_sigma = u[:, ::-1, :]  # Reflect y-coordinate
        v_sigma = -v[:, ::-1, :] # Reflect y-coordinate and negate v
        w_sigma = w[:, ::-1, :]  # Reflect y-coordinate
        
        return u_sigma, v_sigma, w_sigma
    
    def antisymmetric_projector(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply antisymmetric projector P_a = (1/2)(I - σ)
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Antisymmetric components (u_a, v_a, w_a)
        """
        u_sigma, v_sigma, w_sigma = self.bilateral_reflection(u, v, w)
        
        u_a = 0.5 * (u - u_sigma)
        v_a = 0.5 * (v - v_sigma)
        w_a = 0.5 * (w - w_sigma)
        
        return u_a, v_a, w_a
    
    def symmetric_projector(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply symmetric projector P_s = (1/2)(I + σ)
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Symmetric components (u_s, v_s, w_s)
        """
        u_sigma, v_sigma, w_sigma = self.bilateral_reflection(u, v, w)
        
        u_s = 0.5 * (u + u_sigma)
        v_s = 0.5 * (v + v_sigma)
        w_s = 0.5 * (w + w_sigma)
        
        return u_s, v_s, w_s
    
    def decompose(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Dict:
        """
        Perform complete antisymmetric decomposition u = u_a + u_s
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Dictionary containing:
            - antisymmetric: (u_a, v_a, w_a)
            - symmetric: (u_s, v_s, w_s)
            - orthogonality_check: ⟨u_a, u_s⟩
            - antisymmetric_dominance: ||u_a|| / ||u||
            - norms: Various L2 norms
        """
        # Apply projectors
        u_a, v_a, w_a = self.antisymmetric_projector(u, v, w)
        u_s, v_s, w_s = self.symmetric_projector(u, v, w)
        
        # Verify orthogonality ⟨u_a, u_s⟩ = 0
        orthogonality = self.spectral_ops.l2_inner_product(
            u_a, v_a, w_a, u_s, v_s, w_s
        )
        
        # Compute norms
        norm_total = self.spectral_ops.compute_l2_norm(u, v, w)
        norm_antisym = self.spectral_ops.compute_l2_norm(u_a, v_a, w_a)
        norm_sym = self.spectral_ops.compute_l2_norm(u_s, v_s, w_s)
        
        # Antisymmetric dominance ratio
        antisym_dominance = norm_antisym / norm_total if norm_total > 0 else 0
        
        # Verify decomposition completeness: ||u||² = ||u_a||² + ||u_s||²
        norm_check = norm_total**2 - (norm_antisym**2 + norm_sym**2)
        
        self.logger.debug(f"Decomposition: ||u_a||={norm_antisym:.6f}, ||u_s||={norm_sym:.6f}")
        self.logger.debug(f"Orthogonality: {orthogonality:.2e}")
        self.logger.debug(f"Antisymmetric dominance: {antisym_dominance:.3f}")
        
        return {
            'antisymmetric': (u_a, v_a, w_a),
            'symmetric': (u_s, v_s, w_s),
            'orthogonality_check': orthogonality,
            'antisymmetric_dominance': antisym_dominance,
            'norms': {
                'total': norm_total,
                'antisymmetric': norm_antisym,
                'symmetric': norm_sym,
                'completeness_check': norm_check
            }
        }
    
    def decompose_scalar(self, field: np.ndarray) -> Dict:
        """
        Decompose scalar field (e.g., pressure) antisymmetrically
        
        Args:
            field: Scalar field to decompose
            
        Returns:
            Dictionary with antisymmetric and symmetric components
        """
        # Apply σ transformation to scalar field
        field_sigma = field[:, ::-1, :]  # Reflect y-coordinate
        
        # Decompose into antisymmetric and symmetric parts
        field_a = 0.5 * (field - field_sigma)
        field_s = 0.5 * (field + field_sigma)
        
        # Verify orthogonality
        orthogonality = np.mean(field_a * field_s) * (self.spectral_ops.dx)**3
        
        return {
            'antisymmetric': field_a,
            'symmetric': field_s,
            'orthogonality_check': orthogonality
        }
    
    def verify_fundamental_cancellation(self, u_a: np.ndarray, v_a: np.ndarray, w_a: np.ndarray) -> Dict:
        """
        Verify the fundamental cancellation property: ⟨(u_a · ∇)u_a, u_a⟩ = 0
        
        This is the core mathematical discovery that enables relaxed global 
        existence conditions.
        
        Args:
            u_a, v_a, w_a: Antisymmetric velocity components
            
        Returns:
            Dictionary with cancellation analysis
        """
        # Compute advection term (u_a · ∇)u_a
        adv_u, adv_v, adv_w = self.spectral_ops.compute_advection(u_a, v_a, w_a)
        
        # Compute the critical inner product ⟨(u_a · ∇)u_a, u_a⟩
        cancellation = self.spectral_ops.l2_inner_product(
            adv_u, adv_v, adv_w, u_a, v_a, w_a
        )
        
        # Compute energy for relative comparison
        energy_antisym = 0.5 * self.spectral_ops.l2_inner_product(
            u_a, v_a, w_a, u_a, v_a, w_a
        )
        
        # Relative cancellation
        cancellation_relative = abs(cancellation) / energy_antisym if energy_antisym > 0 else 0
        
        # Theoretical threshold verification
        theoretical_bound = 5e-19  # Realistic numerical precision
        verification_passed = abs(cancellation) < theoretical_bound
        
        self.logger.info(f"Fundamental cancellation: {cancellation:.2e}")
        self.logger.info(f"Relative to energy: {cancellation_relative:.2e}")
        self.logger.info(f"Verification: {'PASS' if verification_passed else 'FAIL'}")
        
        return {
            'absolute_cancellation': cancellation,
            'relative_cancellation': cancellation_relative,
            'antisymmetric_energy': energy_antisym,
            'theoretical_bound': theoretical_bound,
            'verification_passed': verification_passed,
            'significance_ratio': abs(cancellation) / theoretical_bound if theoretical_bound > 0 else float('inf')
        }
    
    def compute_antisymmetric_properties(self, u_a: np.ndarray, v_a: np.ndarray, w_a: np.ndarray) -> Dict:
        """
        Compute comprehensive properties of antisymmetric velocity field
        
        Args:
            u_a, v_a, w_a: Antisymmetric velocity components
            
        Returns:
            Dictionary with complete antisymmetric field analysis
        """
        # Basic norms
        l2_norm = self.spectral_ops.compute_l2_norm(u_a, v_a, w_a)
        h1_norm = self.spectral_ops.compute_h1_norm(u_a, v_a, w_a)
        
        # Divergence verification (should be ~0 for incompressible)
        divergence = self.spectral_ops.divergence(u_a, v_a, w_a)
        max_divergence = np.max(np.abs(divergence))
        
        # Fundamental cancellation
        cancellation_analysis = self.verify_fundamental_cancellation(u_a, v_a, w_a)
        
        # Symmetry verification: verify that σ(u_a) = -u_a
        u_a_sigma, v_a_sigma, w_a_sigma = self.bilateral_reflection(u_a, v_a, w_a)
        antisymmetry_error = np.sqrt(np.mean(
            (u_a + u_a_sigma)**2 + (v_a + v_a_sigma)**2 + (w_a + w_a_sigma)**2
        )) * (self.spectral_ops.dx)**(3/2)
        
        return {
            'l2_norm': l2_norm,
            'h1_norm': h1_norm,
            'max_divergence': max_divergence,
            'cancellation': cancellation_analysis,
            'antisymmetry_error': antisymmetry_error,
            'properties_valid': {
                'incompressible': max_divergence < 1e-12,
                'antisymmetric': antisymmetry_error < 1e-14,
                'cancellation': cancellation_analysis['verification_passed']
            }
        }