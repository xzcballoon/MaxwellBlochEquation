# Maxwell-Bloch Equation Solver

A Python implementation for solving the Maxwell-Bloch equations, which describe the interaction between electromagnetic radiation and matter in the semi-classical approximation.

## Overview

The Maxwell-Bloch equations are a set of coupled differential equations fundamental to quantum optics and laser physics. They describe:

1. **Maxwell's Equation**: The propagation of the electromagnetic field through a medium
2. **Bloch Equations**: The quantum mechanical evolution of a two-level atom system

### The Equations

The system consists of three coupled differential equations:

```
dE/dt = -g * p
dp/dt = -γ_⊥ * p + g * E * w
dw/dt = -γ_∥ * (w + 1) - 2 * g * E * p
```

Where:
- **E**: Electric field amplitude
- **p**: Polarization (coherence between atomic states)
- **w**: Population inversion (difference between excited and ground state populations)
- **g**: Coupling strength between field and atoms
- **γ_⊥** (gamma_perp): Transverse relaxation rate (dephasing)
- **γ_∥** (gamma_parallel): Longitudinal relaxation rate (population decay)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import numpy as np
from maxwell_bloch import solve_maxwell_bloch

# Time array
t = np.linspace(0, 50, 1000)

# Initial conditions: [E0, p0, w0]
initial_state = [0.1, 0.5, 0.8]

# Solve the equations
t, E, p, w = solve_maxwell_bloch(
    t, initial_state,
    omega_0=1.0,         # Resonance frequency
    gamma_perp=0.1,      # Transverse relaxation
    gamma_parallel=0.02, # Longitudinal relaxation
    g=0.3                # Coupling strength
)
```

### Object-Oriented Interface

```python
from maxwell_bloch import MaxwellBlochSolver

# Create solver
solver = MaxwellBlochSolver(
    omega_0=1.0,
    gamma_perp=0.1,
    gamma_parallel=0.01,
    g=0.5
)

# Solve
solution = solver.solve(t, initial_state)
E, p, w = solution[:, 0], solution[:, 1], solution[:, 2]
```

### Running Examples

The repository includes several example scenarios:

```bash
python example.py
```

This will generate three examples:
1. **Free Relaxation**: System evolution without external driving
2. **Pulse Propagation**: Electromagnetic pulse interacting with the medium
3. **Rabi Oscillations**: Coherent oscillations in strong coupling regime

Each example produces a visualization saved as a PNG file.

## Physical Interpretation

### Parameters

- **omega_0**: Atomic transition frequency (energy difference between levels)
- **gamma_perp**: Rate at which coherence is lost (dephasing from collisions, etc.)
- **gamma_parallel**: Rate at which excited atoms decay to ground state
- **g**: How strongly the field couples to the atomic transition

### State Variables

- **E(t)**: Electric field envelope amplitude
- **p(t)**: Atomic polarization (off-diagonal density matrix element)
- **w(t)**: Population inversion, ranges from -1 (all ground) to +1 (all excited)

### Applications

These equations are fundamental to understanding:
- Laser dynamics and mode-locking
- Self-induced transparency
- Optical solitons
- Quantum optics phenomena
- Coherent light-matter interactions

## Theory Background

The Maxwell-Bloch equations combine:

1. **Maxwell's wave equation** in the slowly-varying envelope approximation
2. **Optical Bloch equations** for two-level atoms in the rotating wave approximation

This semi-classical treatment treats the electromagnetic field classically while treating the atomic system quantum mechanically. It's valid when:
- The field is strong enough that quantum fluctuations are negligible
- Atoms can be approximated as two-level systems
- The rotating wave approximation is valid (near resonance)

## References

- L. Allen and J. H. Eberly, "Optical Resonance and Two-Level Atoms"
- R. W. Boyd, "Nonlinear Optics"
- M. O. Scully and M. S. Zubairy, "Quantum Optics"

## License

This project is open source and available for educational and research purposes.
