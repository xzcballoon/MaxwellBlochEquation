"""
Unit tests for the Maxwell-Bloch equations solver.
"""

import numpy as np
from maxwell_bloch import MaxwellBlochSolver, solve_maxwell_bloch


def test_solver_initialization():
    """Test that solver can be initialized with default parameters."""
    solver = MaxwellBlochSolver()
    assert solver.omega_0 == 1.0
    assert solver.gamma_perp == 0.1
    assert solver.gamma_parallel == 0.01
    assert solver.g == 0.5
    print("✓ Solver initialization test passed")


def test_solver_custom_parameters():
    """Test that solver accepts custom parameters."""
    solver = MaxwellBlochSolver(omega_0=2.0, gamma_perp=0.2, gamma_parallel=0.02, g=0.3)
    assert solver.omega_0 == 2.0
    assert solver.gamma_perp == 0.2
    assert solver.gamma_parallel == 0.02
    assert solver.g == 0.3
    print("✓ Custom parameters test passed")


def test_equations_shape():
    """Test that equations return correct shape."""
    solver = MaxwellBlochSolver()
    state = [1.0, 0.5, -0.5]
    derivatives = solver.equations(state, 0.0)
    assert derivatives.shape == (3,)
    assert len(derivatives) == 3
    print("✓ Equations shape test passed")


def test_solve_returns_correct_shape():
    """Test that solve returns array with correct dimensions."""
    solver = MaxwellBlochSolver()
    t = np.linspace(0, 10, 100)
    initial_state = [1.0, 0.0, -1.0]
    solution = solver.solve(t, initial_state)
    
    assert solution.shape == (100, 3)
    assert solution.shape[0] == len(t)
    print("✓ Solve shape test passed")


def test_convenience_function():
    """Test the convenience function solve_maxwell_bloch."""
    t = np.linspace(0, 10, 50)
    initial_state = [0.5, 0.2, -0.8]
    
    t_out, E, p, w = solve_maxwell_bloch(t, initial_state)
    
    assert len(t_out) == len(t)
    assert len(E) == len(t)
    assert len(p) == len(t)
    assert len(w) == len(t)
    print("✓ Convenience function test passed")


def test_population_bounds():
    """Test that population inversion stays within physical bounds [-1, 1]."""
    t = np.linspace(0, 50, 500)
    initial_state = [0.1, 0.0, -1.0]  # Start in ground state
    
    t_out, E, p, w = solve_maxwell_bloch(t, initial_state, gamma_perp=0.1, gamma_parallel=0.05, g=0.2)
    
    # Population inversion should stay between -1 and 1 (with some numerical tolerance)
    assert np.all(w >= -1.01), f"Population went below -1: min={np.min(w)}"
    assert np.all(w <= 1.01), f"Population went above 1: max={np.max(w)}"
    print("✓ Population bounds test passed")


def test_relaxation_to_ground_state():
    """Test that system relaxes to ground state without driving field."""
    t = np.linspace(0, 100, 500)
    initial_state = [0.0, 0.1, 0.5]  # Start with some excitation
    
    t_out, E, p, w = solve_maxwell_bloch(
        t, initial_state, 
        gamma_perp=0.1, 
        gamma_parallel=0.05, 
        g=0.2
    )
    
    # System should relax toward w ≈ -1 (ground state)
    assert w[-1] < w[0], "Population inversion should decrease"
    assert w[-1] < 0, "Should relax toward ground state (w < 0)"
    print("✓ Relaxation to ground state test passed")


def test_energy_conservation_weak_relaxation():
    """Test approximate energy conservation with very weak relaxation."""
    t = np.linspace(0, 20, 200)
    initial_state = [1.0, 0.0, -1.0]
    
    # Very weak relaxation
    t_out, E, p, w = solve_maxwell_bloch(
        t, initial_state,
        gamma_perp=0.001,
        gamma_parallel=0.0001,
        g=0.3
    )
    
    # Calculate total "energy" (E^2 + p^2 + w^2)
    energy = E**2 + p**2 + w**2
    
    # Energy should not increase significantly (within 20% due to weak relaxation)
    energy_change = (energy[-1] - energy[0]) / energy[0]
    assert abs(energy_change) < 0.2, f"Energy changed by {energy_change*100:.1f}%"
    print("✓ Energy conservation test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Maxwell-Bloch Solver Tests")
    print("=" * 60)
    print()
    
    test_solver_initialization()
    test_solver_custom_parameters()
    test_equations_shape()
    test_solve_returns_correct_shape()
    test_convenience_function()
    test_population_bounds()
    test_relaxation_to_ground_state()
    test_energy_conservation_weak_relaxation()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
