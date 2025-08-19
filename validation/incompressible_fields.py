#!/usr/bin/env python3
"""
INCOMPRESSIBLE FIELDS MODULE
===========================

Generation of rigorously incompressible velocity fields for validation
of the antisymmetric decomposition paper. This module provides:

1. Vector potential methods for exact divergence-free fields
2. Antisymmetric-dominated field generation
3. Specified H¹ norm targeting
4. Incompressibility verification with machine precision

All generated fields satisfy ∇·u = 0 to machine precision and can be
tuned for antisymmetric dominance ratios required by the paper.

Author: Celso Campos
Date: August 18, 2025
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

class IncompressibleFieldGenerator:
    """
    Generator for incompressible velocity fields using vector potential methods
    
    This class creates velocity fields that are guaranteed to be divergence-free
    (∇·u = 0) through the use of vector potentials: u = ∇ × A.
    Fields can be specifically designed for antisymmetric dominance.
    """
    
    def __init__(self, spectral_ops):
        """
        Initialize field generator
        
        Args:
            spectral_ops: SpectralOperators instance for computations
        """
        self.spectral_ops = spectral_ops
        self.N = spectral_ops.N
        self.L = spectral_ops.L
        self.dx = spectral_ops.dx
        
        # Create coordinate arrays
        x = np.linspace(0, self.L, self.N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')
        
        self.logger = logging.getLogger(__name__)
    
    def create_vector_potential_antisymmetric(self, amplitude: float = 1.0, 
                                            antisym_bias: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create vector potential A designed to produce antisymmetric-dominated fields
        
        The resulting velocity field u = ∇ × A will have enhanced antisymmetric
        components when decomposed under σ(x,y,z) = (x,-y,z).
        
        Args:
            amplitude: Overall amplitude scaling
            antisym_bias: Bias toward antisymmetric components (0.5 = balanced, 1.0 = pure antisym)
            
        Returns:
            Vector potential components (A1, A2, A3)
        """
        X, Y, Z = self.X, self.Y, self.Z
        
        # Design potential components to favor antisymmetric structure
        # A1: Strong antisymmetric component - odd in y
        A1 = amplitude * antisym_bias * (
            0.5 * np.sin(2*np.pi*Y/self.L) * np.cos(np.pi*Z/self.L) +
            0.4 * np.cos(np.pi*X/self.L) * (Y/self.L - 0.5) * np.sin(2*np.pi*Z/self.L)
        )
        
        # A2: Primary antisymmetric generator - strongly odd in y
        A2 = amplitude * antisym_bias * (
            0.6 * np.cos(np.pi*X/self.L) * (Y/self.L - 0.5) * np.sin(np.pi*Z/self.L) +
            0.5 * np.sin(np.pi*X/self.L) * np.sin(3*np.pi*Y/self.L) * np.cos(np.pi*Z/self.L) +
            0.3 * (Y/self.L - 0.5)**3 * np.cos(2*np.pi*X/self.L) * np.sin(np.pi*Z/self.L)
        )
        
        # A3: Mixed component with strong antisymmetric bias
        A3 = amplitude * antisym_bias * (
            0.7 * np.sin(np.pi*X/self.L) * np.sin(2*np.pi*Y/self.L) * np.cos(np.pi*Z/self.L) +
            0.4 * np.cos(2*np.pi*X/self.L) * (Y/self.L - 0.5)**2 * np.sin(np.pi*Z/self.L) +
            0.3 * np.sin(3*np.pi*Y/self.L) * np.cos(np.pi*X/self.L) * np.cos(2*np.pi*Z/self.L)
        )
        
        # Add small symmetric components for realism
        symmetric_amplitude = amplitude * (1 - antisym_bias) * 0.3
        
        A1 += symmetric_amplitude * np.cos(np.pi*X/self.L) * np.cos(np.pi*Y/self.L) * np.sin(np.pi*Z/self.L)
        A2 += symmetric_amplitude * np.sin(np.pi*X/self.L) * np.cos(2*np.pi*Y/self.L) * np.cos(np.pi*Z/self.L)
        A3 += symmetric_amplitude * np.cos(np.pi*X/self.L) * np.sin(np.pi*Y/self.L) * np.cos(2*np.pi*Z/self.L)
        
        self.logger.debug(f"Vector potential created with antisym_bias={antisym_bias}")
        
        return A1, A2, A3
    
    def curl_vector_potential(self, A1: np.ndarray, A2: np.ndarray, A3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute curl of vector potential: u = ∇ × A
        
        This guarantees ∇·u = 0 by the vector identity ∇·(∇ × A) = 0
        
        Args:
            A1, A2, A3: Vector potential components
            
        Returns:
            Velocity field components (u, v, w)
        """
        # Compute gradients of potential components
        dA1_dx, dA1_dy, dA1_dz = self.spectral_ops.gradient(A1)
        dA2_dx, dA2_dy, dA2_dz = self.spectral_ops.gradient(A2)
        dA3_dx, dA3_dy, dA3_dz = self.spectral_ops.gradient(A3)
        
        # Curl: u = ∇ × A
        u = dA3_dy - dA2_dz  # ∂A3/∂y - ∂A2/∂z
        v = dA1_dz - dA3_dx  # ∂A1/∂z - ∂A3/∂x
        w = dA2_dx - dA1_dy  # ∂A2/∂x - ∂A1/∂y
        
        return u, v, w
    
    def create_incompressible_field(self, target_amplitude: float = 1.0,
                                  antisym_dominance: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create incompressible velocity field with specified antisymmetric dominance
        
        Args:
            target_amplitude: Target L2 norm amplitude
            antisym_dominance: Desired antisymmetric dominance ratio (0.5-1.0)
            
        Returns:
            Incompressible velocity field (u, v, w)
        """
        # Create vector potential biased toward antisymmetric structure
        A1, A2, A3 = self.create_vector_potential_antisymmetric(
            amplitude=1.0, antisym_bias=antisym_dominance
        )
        
        # Generate velocity field
        u, v, w = self.curl_vector_potential(A1, A2, A3)
        
        # Scale to target amplitude
        current_norm = self.spectral_ops.compute_l2_norm(u, v, w)
        if current_norm > 0:
            scaling = target_amplitude / current_norm
            u *= scaling
            v *= scaling
            w *= scaling
        
        # Verify incompressibility
        div_u = self.spectral_ops.divergence(u, v, w)
        max_divergence = np.max(np.abs(div_u))
        
        self.logger.info(f"Generated field: ||u||_L2={self.spectral_ops.compute_l2_norm(u, v, w):.6f}")
        self.logger.info(f"Max divergence: {max_divergence:.2e}")
        
        return u, v, w
    
    def create_antisymmetric_dominated_field(self, target_h1_norm: float = 0.673,
                                           target_antisym_ratio: float = 0.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create field with specified H¹ norm and antisymmetric dominance
        
        This method iteratively adjusts the field to meet both the H¹ norm
        requirement and antisymmetric dominance condition from the paper.
        
        Args:
            target_h1_norm: Target H¹ norm (paper uses 0.673)
            target_antisym_ratio: Target antisymmetric dominance (≥0.6 required)
            
        Returns:
            Tuned velocity field (u, v, w)
        """
        max_iterations = 20
        tolerance = 0.01
        
        # Start with very high antisymmetric bias for target ≥0.6
        antisym_bias = min(0.99, target_antisym_ratio + 0.3)
        
        for iteration in range(max_iterations):
            # Generate field with strong antisymmetric bias
            u, v, w = self.create_incompressible_field(
                target_amplitude=target_h1_norm * 0.8,  # Initial estimate
                antisym_dominance=antisym_bias
            )
            
            # Check current properties
            current_h1 = self.spectral_ops.compute_h1_norm(u, v, w)
            
            # Check antisymmetric dominance
            current_antisym_ratio = self._estimate_antisym_dominance(u, v, w)
            
            # Scale to target H¹ norm
            if current_h1 > 0:
                h1_scaling = target_h1_norm / current_h1
                u *= h1_scaling
                v *= h1_scaling
                w *= h1_scaling
                current_h1 = target_h1_norm
            
            self.logger.debug(f"Iteration {iteration+1}: H¹={current_h1:.6f}, antisym≈{current_antisym_ratio:.3f}, bias={antisym_bias:.3f}")
            
            # Check convergence
            h1_error = abs(current_h1 - target_h1_norm) / target_h1_norm
            antisym_satisfied = current_antisym_ratio >= target_antisym_ratio
            
            if h1_error < tolerance and antisym_satisfied:
                self.logger.info(f"Converged in {iteration+1} iterations: antisym={current_antisym_ratio:.3f}")
                break
            
            # Aggressive adjustment for antisymmetric dominance
            if current_antisym_ratio < target_antisym_ratio:
                # Increase bias more aggressively
                antisym_bias = min(0.995, antisym_bias + 0.02)
                self.logger.debug(f"Increasing antisym_bias to {antisym_bias:.3f}")
            
        # If still not achieving target, try pure antisymmetric field
        if iteration == max_iterations - 1:
            self.logger.warning("Creating pure antisymmetric field")
            u, v, w = self._create_pure_antisymmetric_field(target_h1_norm)
            
        return u, v, w
    
    def _estimate_antisym_dominance(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Estimate antisymmetric dominance without full decomposition
        
        This provides a quick estimate based on field symmetry properties.
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Estimated antisymmetric dominance ratio
        """
        # Apply bilateral reflection
        u_sigma = u[:, ::-1, :]
        v_sigma = -v[:, ::-1, :]
        w_sigma = w[:, ::-1, :]
        
        # Antisymmetric part
        u_a = 0.5 * (u - u_sigma)
        v_a = 0.5 * (v - v_sigma)
        w_a = 0.5 * (w - w_sigma)
        
        # Compute norms
        norm_total = self.spectral_ops.compute_l2_norm(u, v, w)
        norm_antisym = self.spectral_ops.compute_l2_norm(u_a, v_a, w_a)
        
        return norm_antisym / norm_total if norm_total > 0 else 0
    
    def _create_pure_antisymmetric_field(self, target_h1_norm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a field that is purely antisymmetric AND exactly incompressible
        
        Uses vector potential method that preserves both antisymmetry and incompressibility
        by construction: u = ∇ × A where A is antisymmetric.
        
        Args:
            target_h1_norm: Target H¹ norm
            
        Returns:
            Pure antisymmetric velocity field that is exactly incompressible
        """
        X, Y, Z = self.X, self.Y, self.Z
        
        # Center Y coordinate for antisymmetry
        Y_centered = Y - self.L/2
        
        # Create antisymmetric vector potential A = (A1, A2, A3)
        # Each component must be odd in y to ensure u = ∇ × A is antisymmetric
        
        # A1: antisymmetric potential (odd in y)
        A1 = Y_centered * np.sin(2*np.pi*X/self.L) * np.cos(np.pi*Z/self.L) + \
             0.3 * Y_centered**3 / (self.L/2)**2 * np.cos(2*np.pi*X/self.L) * np.sin(2*np.pi*Z/self.L)
        
        # A2: antisymmetric potential (odd in y) 
        A2 = 0.8 * Y_centered * np.cos(np.pi*X/self.L) * np.sin(np.pi*Z/self.L) + \
             0.4 * Y_centered**3 / (self.L/2)**2 * np.sin(2*np.pi*X/self.L) * np.cos(2*np.pi*Z/self.L)
        
        # A3: antisymmetric potential (odd in y)
        A3 = 0.6 * Y_centered * np.sin(np.pi*X/self.L) * np.cos(2*np.pi*Z/self.L) + \
             0.25 * Y_centered**3 / (self.L/2)**2 * np.cos(3*np.pi*X/self.L) * np.sin(np.pi*Z/self.L)
        
        # Compute curl to get velocity field: u = ∇ × A
        # This automatically guarantees ∇·u = 0 (vector identity)
        u, v, w = self.curl_vector_potential(A1, A2, A3)
        
        # Verify antisymmetry is preserved
        u_sigma = u[:, ::-1, :]
        v_sigma = -v[:, ::-1, :]
        w_sigma = w[:, ::-1, :]
        
        antisym_error_u = np.max(np.abs(u + u_sigma))
        antisym_error_v = np.max(np.abs(v + v_sigma))
        antisym_error_w = np.max(np.abs(w + w_sigma))
        
        self.logger.debug(f"Antisymmetry errors: u={antisym_error_u:.2e}, v={antisym_error_v:.2e}, w={antisym_error_w:.2e}")
        
        # If antisymmetry is not perfect, apply projector
        if max(antisym_error_u, antisym_error_v, antisym_error_w) > 1e-14:
            u = 0.5 * (u - u_sigma)
            v = 0.5 * (v - v_sigma)
            w = 0.5 * (w - w_sigma)
            
            # Since we modified the field, we need to restore incompressibility
            u, v, w = self._project_incompressible_antisymmetric(u, v, w)
        
        # Scale to target H¹ norm
        current_h1 = self.spectral_ops.compute_h1_norm(u, v, w)
        if current_h1 > 0:
            scaling = target_h1_norm / current_h1
            u *= scaling
            v *= scaling
            w *= scaling
        
        # Final verification
        div_final = self.spectral_ops.divergence(u, v, w)
        max_div_final = np.max(np.abs(div_final))
        
        self.logger.info(f"Pure antisymmetric field: H¹={self.spectral_ops.compute_h1_norm(u, v, w):.6f}, max_div={max_div_final:.2e}")
        
        return u, v, w
    
    def _project_incompressible_antisymmetric(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project to incompressible space while preserving antisymmetry
        
        This iterative method alternates between incompressible projection
        and antisymmetric projection until both conditions are satisfied.
        
        Args:
            u, v, w: Input velocity field
            
        Returns:
            Field that is both incompressible and antisymmetric
        """
        max_iterations = 10
        tolerance = 1e-15
        
        for iteration in range(max_iterations):
            # Step 1: Project to divergence-free
            u, v, w = self.spectral_ops.project_divergence_free(u, v, w)
            
            # Step 2: Project to antisymmetric
            u_sigma = u[:, ::-1, :]
            v_sigma = -v[:, ::-1, :]
            w_sigma = w[:, ::-1, :]
            
            u = 0.5 * (u - u_sigma)
            v = 0.5 * (v - v_sigma)
            w = 0.5 * (w - w_sigma)
            
            # Check convergence
            div_u = self.spectral_ops.divergence(u, v, w)
            max_divergence = np.max(np.abs(div_u))
            
            # Check antisymmetry
            antisym_error = max(
                np.max(np.abs(u + u[:, ::-1, :])),
                np.max(np.abs(v - (-v[:, ::-1, :]))),
                np.max(np.abs(w + w[:, ::-1, :]))
            )
            
            if max_divergence < tolerance and antisym_error < tolerance:
                self.logger.debug(f"Converged in {iteration+1} iterations: div={max_divergence:.2e}, antisym_err={antisym_error:.2e}")
                break
        
        return u, v, w
    
    def create_test_field(self, field_type: str = "taylor_green") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create standard test fields for validation
        
        Args:
            field_type: Type of test field ("taylor_green", "abc_flow", "simple_sine")
            
        Returns:
            Test velocity field (u, v, w)
        """
        X, Y, Z = self.X, self.Y, self.Z
        
        if field_type == "taylor_green":
            # Modified Taylor-Green vortex with antisymmetric enhancement
            u = np.sin(2*np.pi*X/self.L) * np.cos(2*np.pi*Y/self.L) * np.cos(2*np.pi*Z/self.L)
            v = -np.cos(2*np.pi*X/self.L) * np.sin(2*np.pi*Y/self.L) * np.cos(2*np.pi*Z/self.L) * (Y/self.L - 0.5)
            w = 0.5 * np.sin(2*np.pi*X/self.L) * np.sin(2*np.pi*Y/self.L) * np.sin(2*np.pi*Z/self.L)
            
        elif field_type == "abc_flow":
            # ABC flow with antisymmetric modification
            A, B, C = 1.0, 0.8, 0.6
            u = A * np.sin(2*np.pi*Z/self.L) + C * np.cos(2*np.pi*Y/self.L)
            v = B * np.sin(2*np.pi*X/self.L) + A * np.cos(2*np.pi*Z/self.L) * (Y/self.L - 0.5)
            w = C * np.sin(2*np.pi*Y/self.L) + B * np.cos(2*np.pi*X/self.L)
            
        elif field_type == "simple_sine":
            # Simple sine field with built-in antisymmetric structure
            u = np.sin(np.pi*X/self.L) * np.cos(np.pi*Z/self.L)
            v = (Y/self.L - 0.5) * np.cos(np.pi*X/self.L) * np.sin(np.pi*Z/self.L)
            w = -np.sin(np.pi*Y/self.L) * np.sin(np.pi*X/self.L)
            
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        # Project to divergence-free space
        u, v, w = self.spectral_ops.project_divergence_free(u, v, w)
        
        return u, v, w
    
    def verify_field_properties(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Dict:
        """
        Comprehensive verification of field properties
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Dictionary with complete field analysis
        """
        # Basic properties
        l2_norm = self.spectral_ops.compute_l2_norm(u, v, w)
        h1_norm = self.spectral_ops.compute_h1_norm(u, v, w)
        
        # Incompressibility check
        div_u = self.spectral_ops.divergence(u, v, w)
        max_divergence = np.max(np.abs(div_u))
        rms_divergence = np.sqrt(np.mean(div_u**2))
        
        # Energy and enstrophy
        kinetic_energy = 0.5 * l2_norm**2
        enstrophy = self.spectral_ops.compute_enstrophy(u, v, w)
        
        # Antisymmetric properties estimate
        antisym_dominance = self._estimate_antisym_dominance(u, v, w)
        
        # Field regularity
        omega_x, omega_y, omega_z = self.spectral_ops.compute_vorticity(u, v, w)
        vorticity_norm = self.spectral_ops.compute_l2_norm(omega_x, omega_y, omega_z)
        
        return {
            'norms': {
                'l2_norm': l2_norm,
                'h1_norm': h1_norm,
                'vorticity_norm': vorticity_norm
            },
            'incompressibility': {
                'max_divergence': max_divergence,
                'rms_divergence': rms_divergence,
                'is_incompressible': max_divergence < 1e-12
            },
            'energetics': {
                'kinetic_energy': kinetic_energy,
                'enstrophy': enstrophy,
                'energy_to_enstrophy_ratio': kinetic_energy / enstrophy if enstrophy > 0 else float('inf')
            },
            'symmetry': {
                'antisymmetric_dominance': antisym_dominance,
                'satisfies_paper_condition': antisym_dominance >= 0.6
            },
            'regularity': {
                'velocity_to_vorticity_ratio': l2_norm / vorticity_norm if vorticity_norm > 0 else float('inf'),
                'field_is_smooth': vorticity_norm < 100 * l2_norm
            }
        }