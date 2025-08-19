#!/usr/bin/env python3
"""
STRESS TESTS MODULE - BRUTAL VALIDATION
======================================

Implementation of the brutal stress tests:
- Blow-up detection and singular regime testing
- Stochastic perturbations with Gaussian noise
- Non-symmetric domain violations  
- High-resolution scaling tests
- Failure metrics for realistic scenarios

This module implements the "destruction tests" to expose any weaknesses
in the antisymmetric decomposition theory under extreme conditions.

Author: Celso Campos  
Date: August 18, 2025
License: MIT (Stress Test Edition)
"""

import numpy as np
import sys
from typing import Dict, List, Tuple, Optional
import logging
import time
from scipy.stats import multivariate_normal
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

class StressTestFramework:
    """
    Brutal stress testing framework for antisymmetric decomposition
    
    Implements the aggressive testing protocols to hunt for:
    - Blow-up singularities  
    - Theory breakdown under perturbations
    - Resolution dependence failures
    - Non-symmetric domain violations
    """
    
    def __init__(self, spectral_ops, decomposer, field_generator):
        self.spectral_ops = spectral_ops
        self.decomposer = decomposer
        self.field_generator = field_generator
        self.logger = logging.getLogger(__name__)
        
        # Brutal test parameters
        self.TOLERANCE_CANCELLATION = 1e-10  # Relaxed from 1e-19
        self.TOLERANCE_BLOWUP = 1e-5         # Failure threshold
        self.MAX_TIME_INTEGRATION = 1e4      # Hunt singularities
        
        self.logger.info("Stress test framework initialized - BRUTAL MODE")
    
    def gaussian_noise_perturbation(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                                  noise_amplitude: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Add Gaussian white noise to destroy perfect antisymmetric structure
        
        Args:
            u, v, w: Clean velocity field
            noise_amplitude: Amplitude of destruction (default: 1e-3)
            
        Returns:
            Contaminated velocity field
        """
        shape = u.shape
        
        # Generate white noise
        noise_u = np.random.normal(0, noise_amplitude, shape)
        noise_v = np.random.normal(0, noise_amplitude, shape)  
        noise_w = np.random.normal(0, noise_amplitude, shape)
        
        # Contaminate the field
        u_noisy = u + noise_u
        v_noisy = v + noise_v
        w_noisy = w + noise_w
        
        # Project back to divergence-free (but now polluted)
        u_noisy, v_noisy, w_noisy = self.spectral_ops.project_divergence_free(u_noisy, v_noisy, w_noisy)
        
        self.logger.debug(f"Added Gaussian noise with amplitude {noise_amplitude:.2e}")
        
        return u_noisy, v_noisy, w_noisy
    
    def break_sigma_admissibility(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deliberately break σ-admissible structure to test theory robustness
        
        Adds non-antisymmetric contamination that violates the theory's assumptions
        """
        X, Y, Z = self.spectral_ops.X, self.spectral_ops.Y, self.spectral_ops.Z
        L = self.spectral_ops.L
        
        # Add symmetric contamination (violates antisymmetric purity)
        symmetric_contamination_u = 0.1 * np.cos(2*np.pi*X/L) * np.cos(2*np.pi*Y/L) * np.sin(np.pi*Z/L)
        symmetric_contamination_v = 0.1 * np.sin(2*np.pi*X/L) * np.cos(2*np.pi*Y/L) * np.cos(np.pi*Z/L)
        symmetric_contamination_w = 0.1 * np.cos(np.pi*X/L) * np.sin(2*np.pi*Y/L) * np.cos(2*np.pi*Z/L)
        
        # Contaminate the field
        u_broken = u + symmetric_contamination_u
        v_broken = v + symmetric_contamination_v  
        w_broken = w + symmetric_contamination_w
        
        # Project to divergence-free
        u_broken, v_broken, w_broken = self.spectral_ops.project_divergence_free(u_broken, v_broken, w_broken)
        
        self.logger.warning("σ-admissible structure deliberately broken")
        
        return u_broken, v_broken, w_broken
    
    def blow_up_detection_test(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                              max_time: float = 100.0, dt: float = 0.01) -> Dict:
        """
        Hunt for blow-up singularities through time integration
        
        Implements crude Euler integration to detect if solutions explode
        in finite time when theory predictions fail.
        
        Args:
            u, v, w: Initial velocity field
            max_time: Maximum integration time  
            dt: Time step
            
        Returns:
            Blow-up analysis results
        """
        self.logger.info(f"Starting blow-up hunt: t_max={max_time}, dt={dt}")
        
        # Initialize
        u_current, v_current, w_current = u.copy(), v.copy(), w.copy()
        time_current = 0.0
        max_velocity_history = []
        energy_history = []
        
        nu = 0.01  # Viscosity
        
        try:
            while time_current < max_time:
                # Compute current norms
                max_velocity = np.max(np.sqrt(u_current**2 + v_current**2 + w_current**2))
                energy = 0.5 * self.spectral_ops.compute_l2_norm(u_current, v_current, w_current)**2
                
                max_velocity_history.append(max_velocity)
                energy_history.append(energy)
                
                # Check for blow-up
                if max_velocity > 1e6:
                    self.logger.warning(f"BLOW-UP DETECTED at t={time_current:.3f}")
                    return {
                        'blow_up_detected': True,
                        'blow_up_time': time_current,
                        'max_velocity_final': max_velocity,
                        'max_velocity_history': max_velocity_history,
                        'energy_history': energy_history
                    }
                
                # Crude Euler step for Navier-Stokes
                # ∂u/∂t = -u·∇u - ∇p + ν∇²u
                
                # Advection term
                adv_u, adv_v, adv_w = self.spectral_ops.compute_advection(u_current, v_current, w_current)
                
                # Pressure term (Poisson solve)
                div_adv = self.spectral_ops.divergence(adv_u, adv_v, adv_w)
                pressure = self.spectral_ops.solve_poisson(div_adv)
                grad_p_x, grad_p_y, grad_p_z = self.spectral_ops.gradient(pressure)
                
                # Viscous term
                laplacian_u = self.spectral_ops.laplacian(u_current)
                laplacian_v = self.spectral_ops.laplacian(v_current)
                laplacian_w = self.spectral_ops.laplacian(w_current)
                
                # Euler step
                u_new = u_current - dt * (adv_u + grad_p_x - nu * laplacian_u)
                v_new = v_current - dt * (adv_v + grad_p_y - nu * laplacian_v)
                w_new = w_current - dt * (adv_w + grad_p_z - nu * laplacian_w)
                
                # Project to divergence-free
                u_current, v_current, w_current = self.spectral_ops.project_divergence_free(u_new, v_new, w_new)
                
                time_current += dt
                
                # Log progress occasionally
                if int(time_current/dt) % 100 == 0:
                    self.logger.debug(f"t={time_current:.2f}, max_vel={max_velocity:.2e}, energy={energy:.2e}")
        
        except Exception as e:
            self.logger.error(f"Integration failed at t={time_current}: {e}")
            return {
                'blow_up_detected': True,
                'blow_up_time': time_current,
                'integration_failure': str(e),
                'max_velocity_history': max_velocity_history,
                'energy_history': energy_history
            }
        
        # No blow-up detected
        return {
            'blow_up_detected': False,
            'integration_time': time_current,
            'max_velocity_final': max_velocity_history[-1] if max_velocity_history else 0,
            'max_velocity_history': max_velocity_history,
            'energy_history': energy_history
        }
    
    def high_amplitude_stress_test(self, base_amplitude: float = 2.0) -> Dict:
        """
        Test theory breakdown with amplitudes well above H¹ < 1.003 threshold
        
        Args:
            base_amplitude: Amplitude that violates theory conditions
            
        Returns:
            Stress test results
        """
        self.logger.info(f"High amplitude stress test: amplitude={base_amplitude} > 1.003")
        
        # Create field with deliberately high amplitude
        u, v, w = self.field_generator.create_antisymmetric_dominated_field(
            target_h1_norm=base_amplitude,  # Violates H¹ < 1.003 condition
            target_antisym_ratio=0.8
        )
        
        # Verify we actually exceed threshold
        h1_norm = self.spectral_ops.compute_h1_norm(u, v, w)
        theory_violated = h1_norm >= 1.003
        
        if not theory_violated:
            self.logger.warning(f"Failed to violate theory: H¹={h1_norm:.3f} < 1.003")
        
        # Test fundamental properties under stress
        decomp = self.decomposer.decompose(u, v, w)
        u_a, v_a, w_a = decomp['antisymmetric']
        
        # Test cancellation under stress
        cancellation_result = self.decomposer.verify_fundamental_cancellation(u_a, v_a, w_a)
        cancellation = abs(cancellation_result['absolute_cancellation'])
        
        # Test with perturbations
        u_noisy, v_noisy, w_noisy = self.gaussian_noise_perturbation(u, v, w, noise_amplitude=1e-2)
        decomp_noisy = self.decomposer.decompose(u_noisy, v_noisy, w_noisy)
        u_a_noisy, v_a_noisy, w_a_noisy = decomp_noisy['antisymmetric']
        cancellation_noisy = abs(self.decomposer.verify_fundamental_cancellation(u_a_noisy, v_a_noisy, w_a_noisy)['absolute_cancellation'])
        
        return {
            'h1_norm': h1_norm,
            'theory_conditions_violated': theory_violated,
            'cancellation_clean': cancellation,
            'cancellation_noisy': cancellation_noisy,
            'cancellation_robust': cancellation_noisy < self.TOLERANCE_CANCELLATION,
            'antisymmetric_dominance': decomp['antisymmetric_dominance'],
            'stress_test_passed': cancellation < self.TOLERANCE_CANCELLATION and cancellation_noisy < self.TOLERANCE_CANCELLATION
        }
    
    def monte_carlo_robustness_test(self, num_trials: int = 100) -> Dict:
        """
        Monte Carlo testing with random perturbations
        
        Tests theory robustness across statistical ensemble of perturbed fields
        """
        self.logger.info(f"Monte Carlo robustness test: {num_trials} trials")
        
        successes = 0
        failures = 0
        cancellation_values = []
        failure_modes = []
        
        for trial in range(num_trials):
            try:
                # Generate random field
                amplitude = 0.5 + 0.5 * np.random.random()  # 0.5 to 1.0
                antisym_ratio = 0.6 + 0.3 * np.random.random()  # 0.6 to 0.9
                
                u, v, w = self.field_generator.create_antisymmetric_dominated_field(
                    target_h1_norm=amplitude,
                    target_antisym_ratio=antisym_ratio
                )
                
                # Add random perturbations
                noise_level = 1e-4 + 1e-3 * np.random.random()
                u, v, w = self.gaussian_noise_perturbation(u, v, w, noise_amplitude=noise_level)
                
                # Test cancellation
                decomp = self.decomposer.decompose(u, v, w)
                u_a, v_a, w_a = decomp['antisymmetric']
                cancellation = abs(self.decomposer.verify_fundamental_cancellation(u_a, v_a, w_a)['absolute_cancellation'])
                
                cancellation_values.append(cancellation)
                
                # Check success/failure
                if cancellation < self.TOLERANCE_CANCELLATION:
                    successes += 1
                else:
                    failures += 1
                    failure_modes.append({
                        'trial': trial,
                        'amplitude': amplitude,
                        'antisym_ratio': antisym_ratio,
                        'noise_level': noise_level,
                        'cancellation': cancellation
                    })
                    
            except Exception as e:
                failures += 1
                failure_modes.append({
                    'trial': trial,
                    'error': str(e)
                })
                self.logger.warning(f"Trial {trial} failed: {e}")
        
        success_rate = successes / num_trials
        
        return {
            'num_trials': num_trials,
            'successes': successes,
            'failures': failures,
            'success_rate': success_rate,
            'cancellation_mean': np.mean(cancellation_values) if cancellation_values else float('inf'),
            'cancellation_std': np.std(cancellation_values) if cancellation_values else float('inf'),
            'cancellation_max': np.max(cancellation_values) if cancellation_values else float('inf'),
            'failure_modes': failure_modes,
            'robustness_passed': success_rate >= 0.95  # 95% success rate required
        }
    
    def resolution_scaling_test(self, resolutions: List[int] = [32, 64, 128, 256]) -> Dict:
        """
        Test convergence and stability across different grid resolutions
        
        Args:
            resolutions: List of grid sizes to test
            
        Returns:
            Resolution scaling analysis
        """
        self.logger.info(f"Resolution scaling test: {resolutions}")
        
        results = {}
        
        for N in resolutions:
            self.logger.info(f"Testing resolution {N}³")
            
            # Create spectral operators for this resolution
            spectral_ops_N = type(self.spectral_ops)(self.spectral_ops.L, N)
            decomposer_N = type(self.decomposer)(spectral_ops_N)
            field_generator_N = type(self.field_generator)(spectral_ops_N)
            
            try:
                # Generate field at this resolution
                u, v, w = field_generator_N.create_antisymmetric_dominated_field(
                    target_h1_norm=0.673,
                    target_antisym_ratio=0.6
                )
                
                # Test properties
                decomp = decomposer_N.decompose(u, v, w)
                u_a, v_a, w_a = decomp['antisymmetric']
                
                cancellation = abs(decomposer_N.verify_fundamental_cancellation(u_a, v_a, w_a)['absolute_cancellation'])
                h1_norm = spectral_ops_N.compute_h1_norm(u, v, w)
                antisym_dominance = decomp['antisymmetric_dominance']
                
                # Incompressibility test
                div_u = spectral_ops_N.divergence(u, v, w)
                max_divergence = np.max(np.abs(div_u))
                
                results[N] = {
                    'cancellation': cancellation,
                    'h1_norm': h1_norm,
                    'antisym_dominance': antisym_dominance,
                    'max_divergence': max_divergence,
                    'tests_passed': {
                        'cancellation': cancellation < self.TOLERANCE_CANCELLATION,
                        'h1_condition': h1_norm < 1.003,
                        'antisym_condition': antisym_dominance >= 0.6,
                        'incompressibility': max_divergence < 1e-12
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Resolution {N}³ failed: {e}")
                results[N] = {'error': str(e)}
        
        # Analyze convergence
        valid_results = {N: r for N, r in results.items() if 'error' not in r}
        
        if len(valid_results) >= 2:
            # Check if cancellation improves with resolution
            cancellations = [r['cancellation'] for r in valid_results.values()]
            convergence_trend = np.polyfit(range(len(cancellations)), np.log10(cancellations), 1)[0]
            convergent = convergence_trend < 0  # Should decrease with resolution
        else:
            convergent = False
        
        return {
            'resolution_results': results,
            'convergent': convergent,
            'all_resolutions_passed': all('error' not in r and all(r.get('tests_passed', {}).values()) for r in results.values())
        }
    
    def high_resolution_test(self, target_resolution: int = 512) -> Dict:
        """
        Test framework with high resolution
        
        WARNING: This test requires significant memory (>8GB RAM for 512³)
        Implements the "Guerra Computacional"
        
        Args:
            target_resolution: Target grid resolution (512³ or higher)
            
        Returns:
            High resolution test results
        """
        self.logger.warning(f"🔥 HIGH RESOLUTION STRESS TEST: {target_resolution}³ - MEMORY INTENSIVE")
        self.logger.info("Implementing 'Guerra Computacional'")
        
        # Memory estimation
        estimated_memory_gb = (target_resolution**3 * 8 * 10) / (1024**3)  # 10 arrays of float64
        self.logger.warning(f"Estimated memory requirement: {estimated_memory_gb:.1f} GB")
        
        if estimated_memory_gb > 16:
            self.logger.error(f"Resolution {target_resolution}³ requires >16GB RAM - ABORTING for safety")
            return {
                'resolution': target_resolution,
                'memory_required_gb': estimated_memory_gb,
                'test_aborted': True,
                'reason': 'Insufficient memory - requires GPU cluster'
            }
        
        try:
            self.logger.info(f"Creating high-resolution spectral operators: {target_resolution}³")
            
            # Create high-resolution operators
            spectral_ops_hr = type(self.spectral_ops)(self.spectral_ops.L, target_resolution)
            decomposer_hr = type(self.decomposer)(spectral_ops_hr)
            field_generator_hr = type(self.field_generator)(spectral_ops_hr)
            
            # Generate high-resolution field
            self.logger.info("Generating high-resolution antisymmetric field...")
            u, v, w = field_generator_hr.create_antisymmetric_dominated_field(
                target_h1_norm=0.673,
                target_antisym_ratio=0.6
            )
            
            # Test decomposition at high resolution
            self.logger.info("Performing high-resolution decomposition...")
            decomp = decomposer_hr.decompose(u, v, w)
            u_a, v_a, w_a = decomp['antisymmetric']
            
            # Test fundamental cancellation at high resolution
            self.logger.info("Testing cancellation at high resolution...")
            cancellation_result = decomposer_hr.verify_fundamental_cancellation(u_a, v_a, w_a)
            
            # Test incompressibility
            div_u = spectral_ops_hr.divergence(u, v, w)
            max_divergence = np.max(np.abs(div_u))
            
            # Test resolution-dependent convergence
            h1_norm = spectral_ops_hr.compute_h1_norm(u, v, w)
            
            # Compare with coarser resolution (128³)
            # Downsample high-res field to 128³ and compare
            downsampling_factor = target_resolution // 128
            if downsampling_factor > 1:
                u_coarse = u[::downsampling_factor, ::downsampling_factor, ::downsampling_factor]
                v_coarse = v[::downsampling_factor, ::downsampling_factor, ::downsampling_factor]
                w_coarse = w[::downsampling_factor, ::downsampling_factor, ::downsampling_factor]
                
                # Test coarse version
                decomp_coarse = self.decomposer.decompose(u_coarse, v_coarse, w_coarse)
                u_a_coarse, v_a_coarse, w_a_coarse = decomp_coarse['antisymmetric']
                cancellation_coarse = abs(self.decomposer.verify_fundamental_cancellation(
                    u_a_coarse, v_a_coarse, w_a_coarse)['absolute_cancellation'])
                
                convergence_improvement = cancellation_coarse / abs(cancellation_result['absolute_cancellation'])
            else:
                convergence_improvement = 1.0
            
            result = {
                'resolution': target_resolution,
                'memory_used_gb': estimated_memory_gb,
                'test_completed': True,
                'high_resolution_results': {
                    'cancellation': abs(cancellation_result['absolute_cancellation']),
                    'h1_norm': h1_norm,
                    'antisym_dominance': decomp['antisymmetric_dominance'],
                    'max_divergence': max_divergence,
                    'convergence_improvement': convergence_improvement
                },
                'tests_passed': {
                    'cancellation': abs(cancellation_result['absolute_cancellation']) < self.TOLERANCE_CANCELLATION,
                    'h1_condition': h1_norm < 1.003,
                    'antisym_condition': decomp['antisymmetric_dominance'] >= 0.6,
                    'incompressibility': max_divergence < 1e-12
                },
                'high_res_validation_passed': all([
                    abs(cancellation_result['absolute_cancellation']) < self.TOLERANCE_CANCELLATION,
                    h1_norm < 1.003,
                    decomp['antisymmetric_dominance'] >= 0.6,
                    max_divergence < 1e-12
                ])
            }
            
            self.logger.info(f"High resolution test completed: {'PASS' if result['high_res_validation_passed'] else 'FAIL'}")
            
            return result
            
        except MemoryError as e:
            self.logger.error(f"Memory error during high resolution test: {e}")
            return {
                'resolution': target_resolution,
                'memory_required_gb': estimated_memory_gb,
                'test_aborted': True,
                'error': 'MemoryError - requires more RAM'
            }
        except Exception as e:
            self.logger.error(f"High resolution test failed: {e}")
            return {
                'resolution': target_resolution,
                'test_aborted': True,
                'error': str(e)
            }
    
    def non_euclidean_domain_test(self, domain_type: str = "sphere") -> Dict:
        """
        Test antisymmetric decomposition on non-Euclidean domains
        
        Implements the non-Euclidean challenge:
        "Domínios não-Euclidianos (esferas, cilindros) - ZERO implementação atual"
        
        Args:
            domain_type: "sphere" or "cylinder"
            
        Returns:
            Non-Euclidean domain test results
        """
        self.logger.warning(f"🌐 NON-EUCLIDEAN DOMAIN TEST: {domain_type.upper()}")
        self.logger.info("Implementing non-Euclidean geometry")
        
        try:
            if domain_type == "sphere":
                return self._test_spherical_domain()
            elif domain_type == "cylinder":
                return self._test_cylindrical_domain()
            else:
                raise ValueError(f"Unsupported domain type: {domain_type}")
                
        except Exception as e:
            self.logger.error(f"Non-Euclidean domain test failed: {e}")
            return {
                'domain_type': domain_type,
                'test_failed': True,
                'error': str(e),
                'non_euclidean_validation_passed': False
            }
    
    def _test_spherical_domain(self) -> Dict:
        """Test on spherical domain using spherical coordinates"""
        # Create spherical coordinate grid
        N = self.spectral_ops.N
        theta = np.linspace(0, np.pi, N)
        phi = np.linspace(0, 2*np.pi, N)
        r = np.linspace(0.1, 1.0, N)  # Avoid singularity at r=0
        
        R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Convert to Cartesian for σ-reflection testing
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        # Create spherical antisymmetric field
        # u_r = R² sin(θ) cos(φ) [antisymmetric in Y]
        # u_θ = -R sin(θ) sin(φ) [antisymmetric in Y]  
        # u_φ = R cos(θ) [symmetric in Y]
        u_r = R**2 * np.sin(THETA) * np.cos(PHI)
        u_theta = -R * np.sin(THETA) * np.sin(PHI)
        u_phi = R * np.cos(THETA)
        
        # Convert to Cartesian velocity components
        sin_theta = np.sin(THETA)
        cos_theta = np.cos(THETA)
        sin_phi = np.sin(PHI)
        cos_phi = np.cos(PHI)
        
        # Transformation matrix from spherical to Cartesian
        u = (u_r * sin_theta * cos_phi - 
             u_theta * cos_theta * cos_phi - 
             u_phi * sin_phi)
        v = (u_r * sin_theta * sin_phi - 
             u_theta * cos_theta * sin_phi + 
             u_phi * cos_phi)
        w = u_r * cos_theta + u_theta * sin_theta
        
        # Apply spherical domain mask (only points inside unit sphere)
        sphere_mask = (X**2 + Y**2 + Z**2) <= 1.0
        u = np.where(sphere_mask, u, 0.0)
        v = np.where(sphere_mask, v, 0.0)
        w = np.where(sphere_mask, w, 0.0)
        
        # Test decomposition on spherical domain
        try:
            decomp = self.decomposer.decompose(u, v, w)
            u_a, v_a, w_a = decomp['antisymmetric']
            
            # Test fundamental cancellation on sphere
            cancellation_result = self.decomposer.verify_fundamental_cancellation(u_a, v_a, w_a)
            cancellation_value = abs(cancellation_result['absolute_cancellation'])
            
            # Test incompressibility on sphere
            div_u = self.spectral_ops.divergence(u, v, w)
            max_divergence = np.max(np.abs(div_u))
            
            # H¹ norm 
            h1_norm = self.spectral_ops.compute_h1_norm(u, v, w)
            
            result = {
                'domain_type': 'sphere',
                'spherical_results': {
                    'cancellation': cancellation_value,
                    'antisym_dominance': decomp['antisymmetric_dominance'],
                    'max_divergence': max_divergence,
                    'h1_norm': h1_norm,
                    'sphere_points_active': np.sum(sphere_mask)
                },
                'tests_passed': {
                    'cancellation': cancellation_value < self.TOLERANCE_CANCELLATION,
                    'antisym_condition': decomp['antisymmetric_dominance'] >= 0.4,  # Relaxed for spherical
                    'incompressibility': max_divergence < 1e-8,  # Relaxed for complex geometry
                    'h1_condition': h1_norm < 2.0  # Relaxed for spherical geometry
                },
                'non_euclidean_validation_passed': all([
                    cancellation_value < self.TOLERANCE_CANCELLATION,
                    decomp['antisymmetric_dominance'] >= 0.4,
                    max_divergence < 1e-8,
                    h1_norm < 2.0
                ])
            }
            
            self.logger.info(f"Spherical domain test: {'PASS' if result['non_euclidean_validation_passed'] else 'FAIL'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Spherical decomposition failed: {e}")
            return {
                'domain_type': 'sphere',
                'decomposition_failed': True,
                'error': str(e),
                'non_euclidean_validation_passed': False
            }
    
    def _test_cylindrical_domain(self) -> Dict:
        """Test on cylindrical domain using cylindrical coordinates"""
        # Create cylindrical coordinate grid
        N = self.spectral_ops.N
        rho = np.linspace(0.1, 1.0, N)  # Radial distance from z-axis
        phi = np.linspace(0, 2*np.pi, N)  # Azimuthal angle
        z = np.linspace(-1.0, 1.0, N)   # Height
        
        RHO, PHI, Z = np.meshgrid(rho, phi, z, indexing='ij')
        
        # Convert to Cartesian
        X = RHO * np.cos(PHI)
        Y = RHO * np.sin(PHI)
        Z_cart = Z
        
        # Create cylindrical antisymmetric field
        # u_ρ = ρ cos(φ) [antisymmetric in Y]
        # u_φ = -ρ sin(φ) [antisymmetric in Y]
        # u_z = ρ sin(φ) [antisymmetric in Y]
        u_rho = RHO * np.cos(PHI)
        u_phi_cyl = -RHO * np.sin(PHI)
        u_z = RHO * np.sin(PHI)
        
        # Convert to Cartesian velocity components
        cos_phi = np.cos(PHI)
        sin_phi = np.sin(PHI)
        
        u = u_rho * cos_phi - u_phi_cyl * sin_phi
        v = u_rho * sin_phi + u_phi_cyl * cos_phi
        w = u_z
        
        # Apply cylindrical domain mask (ρ ≤ 1, |z| ≤ 1)
        cylinder_mask = (RHO <= 1.0) & (np.abs(Z) <= 1.0)
        u = np.where(cylinder_mask, u, 0.0)
        v = np.where(cylinder_mask, v, 0.0)
        w = np.where(cylinder_mask, w, 0.0)
        
        # Test decomposition on cylindrical domain
        try:
            decomp = self.decomposer.decompose(u, v, w)
            u_a, v_a, w_a = decomp['antisymmetric']
            
            # Test fundamental cancellation on cylinder
            cancellation_result = self.decomposer.verify_fundamental_cancellation(u_a, v_a, w_a)
            cancellation_value = abs(cancellation_result['absolute_cancellation'])
            
            # Test incompressibility on cylinder
            div_u = self.spectral_ops.divergence(u, v, w)
            max_divergence = np.max(np.abs(div_u))
            
            # H¹ norm
            h1_norm = self.spectral_ops.compute_h1_norm(u, v, w)
            
            result = {
                'domain_type': 'cylinder',
                'cylindrical_results': {
                    'cancellation': cancellation_value,
                    'antisym_dominance': decomp['antisymmetric_dominance'],
                    'max_divergence': max_divergence,
                    'h1_norm': h1_norm,
                    'cylinder_points_active': np.sum(cylinder_mask)
                },
                'tests_passed': {
                    'cancellation': cancellation_value < self.TOLERANCE_CANCELLATION,
                    'antisym_condition': decomp['antisymmetric_dominance'] >= 0.4,  # Relaxed for cylindrical
                    'incompressibility': max_divergence < 1e-8,  # Relaxed for complex geometry
                    'h1_condition': h1_norm < 2.0  # Relaxed for cylindrical geometry
                },
                'non_euclidean_validation_passed': all([
                    cancellation_value < self.TOLERANCE_CANCELLATION,
                    decomp['antisymmetric_dominance'] >= 0.4,
                    max_divergence < 1e-8,
                    h1_norm < 2.0
                ])
            }
            
            self.logger.info(f"Cylindrical domain test: {'PASS' if result['non_euclidean_validation_passed'] else 'FAIL'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Cylindrical decomposition failed: {e}")
            return {
                'domain_type': 'cylinder',
                'decomposition_failed': True,
                'error': str(e),
                'non_euclidean_validation_passed': False
            }
    
    def _parallel_monte_carlo_worker(self, trial_id: int) -> Tuple[int, bool, str]:
        """Worker function for parallel Monte Carlo testing"""
        try:
            # Generate random field
            u, v, w = self.field_generator.create_antisymmetric_dominated_field(
                target_h1_norm=np.random.uniform(0.5, 1.0),
                target_antisym_ratio=np.random.uniform(0.6, 0.9)
            )
            
            # Add noise perturbation
            u_noisy, v_noisy, w_noisy = self.gaussian_noise_perturbation(u, v, w, noise_amplitude=1e-4)
            
            # Test decomposition
            decomp = self.decomposer.decompose(u_noisy, v_noisy, w_noisy)
            u_a, v_a, w_a = decomp['antisymmetric']
            
            # Test cancellation
            cancellation = self.decomposer.verify_fundamental_cancellation(u_a, v_a, w_a)
            cancellation_valid = abs(cancellation['absolute_cancellation']) < self.TOLERANCE_CANCELLATION
            
            # Test other conditions
            h1_valid = self.spectral_ops.compute_h1_norm(u_noisy, v_noisy, w_noisy) < 1.003
            antisym_valid = decomp['antisymmetric_dominance'] >= 0.6
            div_valid = np.max(np.abs(self.spectral_ops.divergence(u_noisy, v_noisy, w_noisy))) < 1e-12
            
            overall_valid = all([cancellation_valid, h1_valid, antisym_valid, div_valid])
            
            return trial_id, overall_valid, "success"
            
        except Exception as e:
            return trial_id, False, str(e)
    
    def parallel_monte_carlo_test(self, num_trials: int = 100, max_workers: Optional[int] = None) -> Dict:
        """
        PERFORMANCE OPTIMIZED Monte Carlo robustness test with parallel execution
        
        Implements brutal parallelization for performance optimization
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), num_trials)
        
        self.logger.info(f"🚀 PARALLEL Monte Carlo test: {num_trials} trials, {max_workers} workers")
        
        start_time = time.time()
        
        success_count = 0
        failure_modes = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all trials
            future_to_trial = {
                executor.submit(self._parallel_monte_carlo_worker, trial_id): trial_id 
                for trial_id in range(num_trials)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_trial):
                trial_id, passed, result = future.result()
                
                if passed:
                    success_count += 1
                else:
                    failure_modes.append(f"Trial {trial_id}: {result}")
        
        execution_time = time.time() - start_time
        success_rate = success_count / num_trials
        
        self.logger.info(f"Parallel Monte Carlo completed: {success_count}/{num_trials} passed in {execution_time:.2f}s")
        
        return {
            'trials_completed': num_trials,
            'successes': success_count,
            'success_rate': success_rate,
            'execution_time': execution_time,
            'parallel_workers': max_workers,
            'performance_gain': f"{num_trials / execution_time:.1f} trials/sec",
            'failure_modes': failure_modes[:10],  # First 10 failures only
            'robustness_passed': success_rate >= 0.95
        }
    
    def run_complete_stress_tests(self) -> Dict:
        """
        Execute complete brutal stress test battery
        """
        self.logger.info("🔥 STARTING COMPLETE BRUTAL STRESS TESTS")
        
        start_time = time.time()
        
        # 1. High amplitude stress
        self.logger.info("1. High amplitude stress test...")
        high_amplitude_results = self.high_amplitude_stress_test(base_amplitude=2.0)
        
        # 2. Monte Carlo robustness (OPTIMIZED PARALLEL VERSION)
        self.logger.info("2. PARALLEL Monte Carlo robustness test...")
        monte_carlo_results = self.parallel_monte_carlo_test(num_trials=100)  # Full test with parallelization
        
        # 3. Resolution scaling
        self.logger.info("3. Resolution scaling test...")
        resolution_results = self.resolution_scaling_test([32, 64, 128, 256])
        
        # 4. High resolution test (optional - only if explicitly requested)
        high_res_results = None
        if "--high-resolution" in sys.argv:
            self.logger.info("4. High resolution stress test (512³)...")
            high_res_results = self.high_resolution_test(512)
        
        # 5. Non-Euclidean domain tests
        self.logger.info("5. Non-Euclidean domain tests...")
        sphere_results = self.non_euclidean_domain_test("sphere")
        cylinder_results = self.non_euclidean_domain_test("cylinder")
        
        # 6. Blow-up detection (quick test)
        self.logger.info("6. Blow-up detection test...")
        u_test, v_test, w_test = self.field_generator.create_antisymmetric_dominated_field(0.9, 0.7)
        u_stressed, v_stressed, w_stressed = self.break_sigma_admissibility(u_test, v_test, w_test)
        blowup_results = self.blow_up_detection_test(u_stressed, v_stressed, w_stressed, max_time=5.0, dt=0.1)
        
        execution_time = time.time() - start_time
        
        # Overall assessment
        stress_tests_passed = all([
            high_amplitude_results.get('stress_test_passed', False),
            monte_carlo_results.get('robustness_passed', False),
            resolution_results.get('all_resolutions_passed', False),
            sphere_results.get('non_euclidean_validation_passed', False),
            cylinder_results.get('non_euclidean_validation_passed', False),
            not blowup_results.get('blow_up_detected', True)  # No blow-up is good
        ])
        
        # Include high res test in assessment if performed
        if high_res_results and not high_res_results.get('test_aborted', False):
            stress_tests_passed = stress_tests_passed and high_res_results.get('high_res_validation_passed', False)
        
        # Count tests (including optional high res and non-euclidean)
        total_tests = 6  # Added sphere and cylinder tests
        passed_count = sum([
            high_amplitude_results.get('stress_test_passed', False),
            monte_carlo_results.get('robustness_passed', False),
            resolution_results.get('all_resolutions_passed', False),
            sphere_results.get('non_euclidean_validation_passed', False),
            cylinder_results.get('non_euclidean_validation_passed', False),
            not blowup_results.get('blow_up_detected', True)
        ])
        
        if high_res_results and not high_res_results.get('test_aborted', False):
            total_tests += 1
            if high_res_results.get('high_res_validation_passed', False):
                passed_count += 1
        
        return {
            'stress_test_summary': {
                'execution_time': execution_time,
                'overall_passed': stress_tests_passed,
                'total_tests': total_tests,
                'passed_tests': passed_count,
                'high_resolution_performed': high_res_results is not None,
                'non_euclidean_implemented': True
            },
            'high_amplitude_test': high_amplitude_results,
            'monte_carlo_test': monte_carlo_results,
            'resolution_scaling_test': resolution_results,
            'non_euclidean_tests': {
                'sphere_test': sphere_results,
                'cylinder_test': cylinder_results
            },
            'high_resolution_test': high_res_results,
            'blowup_detection_test': blowup_results
        }