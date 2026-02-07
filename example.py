"""
Example usage of the Maxwell-Bloch equations solver.

This script demonstrates several scenarios:
1. Free relaxation (no external field)
2. Pulse propagation through the medium
3. Rabi oscillations
"""

import numpy as np
import matplotlib.pyplot as plt
from maxwell_bloch import solve_maxwell_bloch


def example_free_relaxation():
    """
    Example 1: Free relaxation of the system.
    
    Starting with some initial excitation, the system relaxes back to
    the ground state through spontaneous emission and dephasing.
    """
    print("Example 1: Free relaxation")
    print("-" * 50)
    
    # Time span
    t = np.linspace(0, 50, 1000)
    
    # Initial conditions: small field, some polarization, inverted population
    initial_state = [0.1, 0.5, 0.8]
    
    # Physical parameters
    omega_0 = 1.0  # Resonance frequency
    gamma_perp = 0.1  # Transverse relaxation
    gamma_parallel = 0.02  # Longitudinal relaxation
    g = 0.3  # Coupling strength
    
    # Solve
    t, E, p, w = solve_maxwell_bloch(
        t, initial_state, omega_0, gamma_perp, gamma_parallel, g
    )
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(t, E, 'b-', linewidth=2)
    axes[0].set_ylabel('Electric Field E', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, p, 'r-', linewidth=2)
    axes[1].set_ylabel('Polarization p', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, w, 'g-', linewidth=2)
    axes[2].set_ylabel('Population Inversion w', fontsize=12)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Free Relaxation in Maxwell-Bloch System', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('free_relaxation.png', dpi=150, bbox_inches='tight')
    print("Figure saved as 'free_relaxation.png'")
    plt.close()
    
    print(f"Initial population inversion: {initial_state[2]:.3f}")
    print(f"Final population inversion: {w[-1]:.3f}")
    print()


def example_pulse_propagation():
    """
    Example 2: Pulse propagation through the medium.
    
    An electromagnetic pulse enters the medium and interacts with the atoms,
    causing changes in both the field and the atomic states.
    """
    print("Example 2: Pulse propagation")
    print("-" * 50)
    
    # Time span
    t = np.linspace(0, 30, 1000)
    
    # Initial conditions: medium in ground state
    initial_state = [1.0, 0.0, -1.0]  # w=-1 means all atoms in ground state
    
    # Physical parameters
    omega_0 = 1.0
    gamma_perp = 0.05
    gamma_parallel = 0.01
    g = 0.4
    
    # Solve
    t, E, p, w = solve_maxwell_bloch(
        t, initial_state, omega_0, gamma_perp, gamma_parallel, g
    )
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(t, E, 'b-', linewidth=2)
    axes[0].set_ylabel('Electric Field E', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, p, 'r-', linewidth=2)
    axes[1].set_ylabel('Polarization p', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, w, 'g-', linewidth=2)
    axes[2].set_ylabel('Population Inversion w', fontsize=12)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Pulse Propagation in Maxwell-Bloch System', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pulse_propagation.png', dpi=150, bbox_inches='tight')
    print("Figure saved as 'pulse_propagation.png'")
    plt.close()
    
    print(f"Initial field amplitude: {initial_state[0]:.3f}")
    print(f"Maximum polarization: {np.max(np.abs(p)):.3f}")
    print(f"Maximum population inversion: {np.max(w):.3f}")
    print()


def example_rabi_oscillations():
    """
    Example 3: Rabi oscillations with strong coupling.
    
    When the coupling is strong and relaxation is weak, the system exhibits
    coherent Rabi oscillations between ground and excited states.
    """
    print("Example 3: Rabi oscillations")
    print("-" * 50)
    
    # Time span
    t = np.linspace(0, 100, 2000)
    
    # Initial conditions: strong field, no polarization, ground state
    initial_state = [2.0, 0.0, -1.0]
    
    # Physical parameters - weak relaxation for coherent oscillations
    omega_0 = 1.0
    gamma_perp = 0.01  # Very weak dephasing
    gamma_parallel = 0.005  # Very weak population relaxation
    g = 0.2  # Moderate coupling
    
    # Solve
    t, E, p, w = solve_maxwell_bloch(
        t, initial_state, omega_0, gamma_perp, gamma_parallel, g
    )
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(t, E, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Electric Field E', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, p, 'r-', linewidth=1.5)
    axes[1].set_ylabel('Polarization p', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, w, 'g-', linewidth=1.5)
    axes[2].set_ylabel('Population Inversion w', fontsize=12)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylim([-1.2, 1.2])
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Rabi Oscillations in Maxwell-Bloch System', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rabi_oscillations.png', dpi=150, bbox_inches='tight')
    print("Figure saved as 'rabi_oscillations.png'")
    plt.close()
    
    # Count oscillations in population inversion
    zero_crossings = np.where(np.diff(np.sign(w)))[0]
    num_oscillations = len(zero_crossings) / 2
    print(f"Approximate number of Rabi oscillations: {num_oscillations:.1f}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Maxwell-Bloch Equations - Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    example_free_relaxation()
    example_pulse_propagation()
    example_rabi_oscillations()
    
    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)
