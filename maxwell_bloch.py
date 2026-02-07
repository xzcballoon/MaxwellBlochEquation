"""
Maxwell-Bloch Equations Solver

This module implements a numerical solver for the Maxwell-Bloch equations,
which describe the interaction between electromagnetic radiation and matter
in the semi-classical approximation.

The equations consist of:
1. Maxwell equation for the electric field propagation
2. Bloch equations for the quantum mechanical state of the medium
   - Population inversion (w)
   - Polarization (p)
"""

import numpy as np
from scipy.integrate import odeint


class MaxwellBlochSolver:
    """
    Solver for the Maxwell-Bloch equations.
    
    The system describes the propagation of an electromagnetic field through
    a resonant medium with two-level atoms.
    
    Parameters
    ----------
    omega_0 : float
        Resonance frequency of the two-level atoms (rad/s)
    gamma_perp : float
        Transverse relaxation rate (dephasing rate) (1/s)
    gamma_parallel : float
        Longitudinal relaxation rate (population relaxation) (1/s)
    g : float
        Coupling strength between field and atoms (rad/s per field unit)
    """
    
    def __init__(self, omega_0=1.0, gamma_perp=0.1, gamma_parallel=0.01, g=0.5):
        self.omega_0 = omega_0  # Stored for future extensions (detuning effects)
        self.gamma_perp = gamma_perp
        self.gamma_parallel = gamma_parallel
        self.g = g
        
        # Note: In the current rotating wave approximation (RWA) and
        # rotating frame formulation, omega_0 is transformed away.
        # This parameter is kept for potential future extensions to
        # include detuning effects: delta = omega_laser - omega_0
    
    def equations(self, state, t, E_ext=None):
        """
        Maxwell-Bloch differential equations.
        
        Parameters
        ----------
        state : array-like
            Current state [E, p, w] where:
            E : electric field amplitude
            p : polarization (imaginary part of coherence)
            w : population inversion
        t : float
            Current time
        E_ext : callable or None
            External driving field as function of time
            
        Returns
        -------
        derivatives : array
            Time derivatives [dE/dt, dp/dt, dw/dt]
        """
        E, p, w = state
        
        # External driving field
        if E_ext is not None:
            E_drive = E_ext(t)
        else:
            E_drive = 0.0
        
        # Maxwell equation for electric field
        # dE/dt = -g * p + damping terms
        dE_dt = -self.g * p
        
        # Bloch equations
        # Polarization dynamics (transverse component)
        dp_dt = -self.gamma_perp * p + self.g * E * w
        
        # Population inversion dynamics
        dw_dt = -self.gamma_parallel * (w + 1) - 2 * self.g * E * p
        
        return np.array([dE_dt, dp_dt, dw_dt])
    
    def solve(self, t_span, initial_state, E_ext=None):
        """
        Solve the Maxwell-Bloch equations.
        
        Parameters
        ----------
        t_span : array-like
            Time points at which to compute the solution
        initial_state : array-like
            Initial state [E0, p0, w0]
        E_ext : callable or None
            External driving field as function of time
            
        Returns
        -------
        solution : ndarray
            Solution array with shape (len(t_span), 3)
            Columns are [E(t), p(t), w(t)]
        """
        solution = odeint(self.equations, initial_state, t_span, args=(E_ext,))
        return solution


def solve_maxwell_bloch(t_span, initial_state, omega_0=1.0, gamma_perp=0.1, 
                         gamma_parallel=0.01, g=0.5, E_ext=None):
    """
    Convenience function to solve Maxwell-Bloch equations.
    
    Parameters
    ----------
    t_span : array-like
        Time points for the solution
    initial_state : array-like
        Initial conditions [E0, p0, w0]
    omega_0 : float, optional
        Resonance frequency (default: 1.0)
    gamma_perp : float, optional
        Transverse relaxation rate (default: 0.1)
    gamma_parallel : float, optional
        Longitudinal relaxation rate (default: 0.01)
    g : float, optional
        Coupling strength (default: 0.5)
    E_ext : callable or None, optional
        External driving field (default: None)
        
    Returns
    -------
    t : ndarray
        Time points
    E : ndarray
        Electric field values
    p : ndarray
        Polarization values
    w : ndarray
        Population inversion values
    """
    solver = MaxwellBlochSolver(omega_0, gamma_perp, gamma_parallel, g)
    solution = solver.solve(t_span, initial_state, E_ext)
    
    return t_span, solution[:, 0], solution[:, 1], solution[:, 2]
