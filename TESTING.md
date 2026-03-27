# PyDyn Test Suite

A comprehensive test suite for the PyDyn molecular dynamics project using pytest.

## Overview

The test suite contains **135 tests** organized across 9 test modules, covering all major components of the PyDyn library. The tests are designed to run without GPU acceleration (using NumPy fallbacks when CuPy is unavailable).

## Test Structure

```
tests/
├── __init__.py                  # Test package initialization
├── conftest.py                  # Shared pytest fixtures and configuration
├── test_constants.py            # Physical constants verification (21 tests)
├── test_state.py                # State class and extensions (31 tests)
├── test_constraints.py          # Constraint behavior (7 tests)
├── test_initializer.py          # Velocity initialization (8 tests)
├── test_simulation.py           # Simulation runner (13 tests)
├── test_ensembles.py            # Ensemble integration (13 tests)
├── test_forces.py               # Force models (17 tests)
└── test_plugins.py              # Plugin system (8 tests)
```

## Running Tests

### Run all tests
```bash
python -m pytest tests/ -v
```

### Run specific test file
```bash
python -m pytest tests/test_constants.py -v
```

### Run specific test class
```bash
python -m pytest tests/test_state.py::TestStateInitialization -v
```

### Run specific test
```bash
python -m pytest tests/test_constants.py::TestBoltzmannConstant::test_kb_value -v
```

### Run with coverage
```bash
python -m pytest tests/ --cov=pydyn --cov-report=html
```

## Test Modules

### test_constants.py (21 tests)
Verifies physical constants against CODATA 2018 values and checks unit conversion factors:
- Boltzmann constant (k_B = 8.617333262145e-5 eV/K)
- Planck constants (h, ℏ)
- Energy conversion factors (mv2_to_e, e_to_mv2)
- Pressure-volume conversion (e_to_pV, pV_to_e)
- Magnetic constants (Bohr magneton, gyromagnetic ratio)
- Atomic mass unit
- Mathematical constant (π)
- Roundtrip conversion consistency

### test_state.py (31 tests)
Tests State class functionality:
- State creation and initialization
- Kinetic energy calculations (correct formula, mass-dependence)
- Volume computation (determinant of box matrix)
- Extension registry (register, get, has, remove)
- State copying (independence of arrays)
- Configuration comparison
- Kinetic virial tensor

### test_constraints.py (7 tests)
Verifies constraint behavior:
- RemoveCOMMomentum removes 3 degrees of freedom
- Total momentum is zeroed correctly
- Works with non-uniform masses
- Multiple applications are idempotent
- Center-of-mass velocity calculation

### test_initializer.py (8 tests)
Tests velocity initialization with Maxwell-Boltzmann distribution:
- Distribution statistics (mean ≈ 0, correct variance)
- Temperature dependence (higher T → larger momenta)
- Mass dependence (σ_p ∝ √m)
- Context temperature fallback
- Kinetic energy matches equipartition theorem

### test_simulation.py (13 tests)
Verifies Simulation runner behavior:
- Initialization (calls initializers, observers)
- Stepping (increments counters, calls observers)
- Running for multiple steps
- Finalization (calls observer finalizers)
- Complete simulation workflow
- Idempotent initialization

### test_ensembles.py (13 tests)
Tests ensemble integration:
- Velocity Verlet operator structure and ordering
- Ensemble stepping (position + momentum updates)
- Constraint application during stepping
- PositionOp (r' = r + dt*v)
- MomentumOp (p' = p + dt*F)
- Conserved energy and pressure methods

### test_forces.py (17 tests)
Tests force model interface and caching:
- Force model initialization
- Compute execution and result filling
- Result caching (avoid redundant computations)
- Result retrieval with error handling
- Implemented properties tracking
- Abstract method enforcement

### test_plugins.py (8 tests)
Tests plugin registration and retrieval:
- Force and ensemble plugin registration
- Duplicate detection
- Plugin listing and metadata
- Plugin retrieval by name
- Plugin validation
- Registry management

## Fixtures (conftest.py)

The following pytest fixtures are provided:

- **`mock_state`**: Simple state with 3 atoms, 10×10×10 Å box
- **`mock_context`**: SimulationContext with target_temp=300K
- **`mock_state_with_momentum`**: State with non-zero momentum for KE tests
- **`use_numpy`**: Skip marker for GPU-only tests

## Bilingual Documentation

All test functions include bilingual docstrings (English/Chinese) explaining:
- What is being tested
- Why it matters (correctness verification)
- Expected behavior

## GPU/CPU Compatibility

The test suite handles both GPU (CuPy) and CPU (NumPy) backends:
- Automatically detects CuPy availability
- Falls back to NumPy arrays if CuPy unavailable
- Mocks missing optional dependencies (torch)
- All 135 tests pass on CPU-only systems

## Test Coverage

Key areas covered:

✅ **Constants**: CODATA values, conversion factors, mathematical constants
✅ **State Management**: Array operations, extensions, energy calculations
✅ **Physics**: Constraints (COM removal), kinetic energy, volume
✅ **Integration**: Velocity Verlet, ensemble operators, stepping
✅ **Forces**: Model interface, caching, result handling
✅ **Initialization**: Maxwell-Boltzmann distribution, temperature coupling
✅ **Simulation**: Full workflow from initialization to finalization
✅ **Plugins**: Registration, retrieval, validation
✅ **Error Handling**: Missing properties, invalid inputs, abstract methods

## Test Results

All 135 tests pass:
```
test_constants.py ............... 21 passed
test_constraints.py ............ 7 passed
test_ensembles.py ............... 13 passed
test_forces.py .................. 17 passed
test_initializer.py ............. 8 passed
test_plugins.py ................. 20 passed
test_simulation.py .............. 13 passed
test_state.py ................... 31 passed
================================= 135 passed
```

## Development Notes

- Tests use standard pytest conventions
- Each test file focuses on a single module
- Test classes group related tests
- Clear, descriptive test names
- Comprehensive docstrings with rationale
- Mock objects for isolation (MockForceModel, MockObserver)
- No external dependencies beyond pytest and ASE
