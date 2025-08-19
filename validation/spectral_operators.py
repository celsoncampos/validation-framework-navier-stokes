#!/usr/bin/env python3
"""
SPECTRAL OPERATORS MODULE
========================

High-precision spectral operators for validation of the antisymmetric 
decomposition paper. This module implements FFT-based differential operators
with machine precision accuracy for:

1. Gradient computations
2. Divergence calculations  
3. Laplacian operations
4. Advection terms
5. Inner products and norms

All operations use periodic boundary conditions with spectral accuracy.

Author: Celso Campos
Date: August 18, 2025
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging

class SpectralOperators:
    """
    High-precision spectral operators using FFT
    
    This class provides spectrally accurate differential operators for
    validating the mathematical claims in the antisymmetric decomposition paper.
    All operations maintain machine precision where theoretically possible.
    """
    
    def __init__(self, domain_size: float, grid_points: int):
        """
        Initialize spectral operators for periodic domain
        
        Args:
            domain_size: Size of cubic domain [0, L]³
            grid_points: Number of grid points per dimension
        """
        self.L = domain_size
        self.N = grid_points
        self.dx = domain_size / grid_points
        
        # Wavenumber arrays for spectral derivatives
        k = np.fft.fftfreq(grid_points, d=self.dx) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        
        # Laplacian operator in Fourier space
        self.k_squared = self.kx**2 + self.ky**2 + self.kz**2
        
        # Avoid division by zero for Poisson solving
        self.k_squared_safe = self.k_squared.copy()
        self.k_squared_safe[0, 0, 0] = 1.0
        
        # Integration weight for L2 inner products
        self.integration_weight = self.dx**3
        
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Spectral operators initialized: L={domain_size}, N={grid_points}")
    
    def gradient(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectral gradient ∇f with machine precision
        
        Args:
            field: Input scalar field
            
        Returns:
            Tuple (∂f/∂x, ∂f/∂y, ∂f/∂z)
        """
        field_hat = np.fft.fftn(field)
        
        # Spectral derivatives
        dfdx_hat = 1j * self.kx * field_hat
        dfdy_hat = 1j * self.ky * field_hat
        dfdz_hat = 1j * self.kz * field_hat
        
        # Transform back to physical space
        dfdx = np.real(np.fft.ifftn(dfdx_hat))
        dfdy = np.real(np.fft.ifftn(dfdy_hat))
        dfdz = np.real(np.fft.ifftn(dfdz_hat))
        
        return dfdx, dfdy, dfdz
    
    def divergence(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute spectral divergence ∇·u
        
        Args:
            u, v, w: Vector field components
            
        Returns:
            Divergence field
        """
        dudx, _, _ = self.gradient(u)
        _, dvdy, _ = self.gradient(v)
        _, _, dwdz = self.gradient(w)
        
        return dudx + dvdy + dwdz
    
    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute spectral Laplacian Δf
        
        Args:
            field: Input scalar field
            
        Returns:
            Laplacian of field
        """
        field_hat = np.fft.fftn(field)
        laplacian_hat = -self.k_squared * field_hat
        return np.real(np.fft.ifftn(laplacian_hat))
    
    def compute_advection(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute advection term (u·∇)u with spectral precision
        
        This is critical for the fundamental cancellation property verification.
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Tuple of advection components ((u·∇)u_x, (u·∇)u_y, (u·∇)u_z)
        """
        # Gradients of velocity components
        dudx, dudy, dudz = self.gradient(u)
        dvdx, dvdy, dvdz = self.gradient(v)
        dwdx, dwdy, dwdz = self.gradient(w)
        
        # Advection terms: (u·∇)u = u∂u/∂x + v∂u/∂y + w∂u/∂z
        advection_u = u * dudx + v * dudy + w * dudz
        advection_v = u * dvdx + v * dvdy + w * dvdz
        advection_w = u * dwdx + v * dwdy + w * dwdz
        
        return advection_u, advection_v, advection_w
    
    def l2_inner_product(self, u1: np.ndarray, v1: np.ndarray, w1: np.ndarray,
                        u2: np.ndarray, v2: np.ndarray, w2: np.ndarray) -> float:
        """
        Compute L2 inner product ⟨u1, u2⟩ = ∫ u1·u2 dx
        
        Args:
            u1, v1, w1: First vector field
            u2, v2, w2: Second vector field
            
        Returns:
            L2 inner product value
        """
        return np.sum(u1 * u2 + v1 * v2 + w1 * w2) * self.integration_weight
    
    def compute_l2_norm(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Compute L2 norm ||u||_L2 = sqrt(⟨u, u⟩)
        
        Args:
            u, v, w: Vector field components
            
        Returns:
            L2 norm
        """
        return np.sqrt(self.l2_inner_product(u, v, w, u, v, w))
    
    def compute_h1_norm(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Compute H1 norm ||u||_H1 = sqrt(||u||²_L2 + ||∇u||²_L2)
        
        Args:
            u, v, w: Vector field components
            
        Returns:
            H1 norm
        """
        # L2 norm squared
        l2_norm_sq = np.sum(u**2 + v**2 + w**2) * self.integration_weight
        
        # Gradient norms squared
        dudx, dudy, dudz = self.gradient(u)
        dvdx, dvdy, dvdz = self.gradient(v)
        dwdx, dwdy, dwdz = self.gradient(w)
        
        grad_norm_sq = np.sum(
            dudx**2 + dudy**2 + dudz**2 +
            dvdx**2 + dvdy**2 + dvdz**2 +
            dwdx**2 + dwdy**2 + dwdz**2
        ) * self.integration_weight
        
        return np.sqrt(l2_norm_sq + grad_norm_sq)
    
    def solve_poisson(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation Δu = rhs with periodic boundary conditions
        
        Args:
            rhs: Right-hand side
            
        Returns:
            Solution u
        """
        rhs_hat = np.fft.fftn(rhs)
        
        # Solve in Fourier space: -k²û = rhs_hat
        u_hat = -rhs_hat / self.k_squared_safe
        u_hat[0, 0, 0] = 0  # Fix DC component to zero (mean = 0)
        
        return np.real(np.fft.ifftn(u_hat))
    
    def project_divergence_free(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project vector field to divergence-free space using Helmholtz decomposition
        
        Solves: u = u_div_free + ∇φ where ∇·u_div_free = 0
        
        Args:
            u, v, w: Input vector field
            
        Returns:
            Divergence-free projection (u_df, v_df, w_df)
        """
        # Compute current divergence
        div_u = self.divergence(u, v, w)
        
        # Solve Poisson equation for potential: Δφ = ∇·u
        phi = self.solve_poisson(div_u)
        
        # Compute gradient of potential
        grad_phi_x, grad_phi_y, grad_phi_z = self.gradient(phi)
        
        # Subtract gradient to make divergence-free
        u_df = u - grad_phi_x
        v_df = v - grad_phi_y
        w_df = w - grad_phi_z
        
        return u_df, v_df, w_df
    
    def compute_vorticity(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute vorticity ω = ∇ × u
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Vorticity components (ω_x, ω_y, ω_z)
        """
        dudx, dudy, dudz = self.gradient(u)
        dvdx, dvdy, dvdz = self.gradient(v)
        dwdx, dwdy, dwdz = self.gradient(w)
        
        omega_x = dwdy - dvdz
        omega_y = dudz - dwdx
        omega_z = dvdx - dudy
        
        return omega_x, omega_y, omega_z
    
    def compute_enstrophy(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Compute enstrophy (0.5 * ||ω||²_L2)
        
        Args:
            u, v, w: Velocity field components
            
        Returns:
            Enstrophy value
        """
        omega_x, omega_y, omega_z = self.compute_vorticity(u, v, w)
        return 0.5 * (np.sum(omega_x**2 + omega_y**2 + omega_z**2) * self.integration_weight)
    
    def verify_spectral_accuracy(self, test_function: str = "sine") -> Dict:
        """
        Verify spectral accuracy by testing on known analytical functions
        
        Args:
            test_function: Type of test function ("sine", "gaussian", "polynomial")
            
        Returns:
            Dictionary with accuracy analysis
        """
        x = np.linspace(0, self.L, self.N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        
        if test_function == "sine":
            # Test function: f = sin(2πx/L) * cos(2πy/L) * sin(2πz/L)
            f = np.sin(2*np.pi*X/self.L) * np.cos(2*np.pi*Y/self.L) * np.sin(2*np.pi*Z/self.L)
            
            # Analytical gradient
            dfdx_exact = (2*np.pi/self.L) * np.cos(2*np.pi*X/self.L) * np.cos(2*np.pi*Y/self.L) * np.sin(2*np.pi*Z/self.L)
            dfdy_exact = -(2*np.pi/self.L) * np.sin(2*np.pi*X/self.L) * np.sin(2*np.pi*Y/self.L) * np.sin(2*np.pi*Z/self.L)
            dfdz_exact = (2*np.pi/self.L) * np.sin(2*np.pi*X/self.L) * np.cos(2*np.pi*Y/self.L) * np.cos(2*np.pi*Z/self.L)
            
            # Analytical Laplacian
            laplacian_exact = -3 * (2*np.pi/self.L)**2 * f
            
        else:
            raise ValueError(f"Test function '{test_function}' not implemented")
        
        # Compute numerical derivatives
        dfdx_num, dfdy_num, dfdz_num = self.gradient(f)
        laplacian_num = self.laplacian(f)
        
        # Compute errors
        grad_error_x = np.max(np.abs(dfdx_num - dfdx_exact))
        grad_error_y = np.max(np.abs(dfdy_num - dfdy_exact))
        grad_error_z = np.max(np.abs(dfdz_num - dfdz_exact))
        laplacian_error = np.max(np.abs(laplacian_num - laplacian_exact))
        
        # Check if errors are within machine precision expectations
        tolerance = 1e-14  # Near machine precision for double precision
        
        return {
            'test_function': test_function,
            'grid_points': self.N,
            'errors': {
                'gradient_x': grad_error_x,
                'gradient_y': grad_error_y,
                'gradient_z': grad_error_z,
                'laplacian': laplacian_error
            },
            'accuracy_check': {
                'gradient_x_pass': grad_error_x < tolerance,
                'gradient_y_pass': grad_error_y < tolerance,
                'gradient_z_pass': grad_error_z < tolerance,
                'laplacian_pass': laplacian_error < tolerance
            },
            'overall_accuracy': all([
                grad_error_x < tolerance,
                grad_error_y < tolerance,
                grad_error_z < tolerance,
                laplacian_error < tolerance
            ])
        }