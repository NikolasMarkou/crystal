# Crystal

Python library for creating and manipulating N-dimensional simplexes as a neural network weight initialization method.

## Quick Reference

```bash
make init     # Install dependencies
make test     # Run tests (pytest -sv)
make build    # Build package
make install  # Install package
```

## Project Structure

```
crystal/
├── crystal/              # Main package
│   ├── simplex.py        # Simplex creation and Simplex class
│   ├── rotation_matrix.py # N-dimensional rotation matrices
│   └── distributions.py   # Wigner semicircle PDF functions
├── tests/                # pytest test suite (62 tests, 1D-1000D coverage)
└── main.py               # Demo script
```

## Core API

```python
import crystal
import numpy as np

# Create simplex matrix (dimensions+1 points in N-dimensions)
matrix = crystal.create_simplex_matrix(dimensions=100, distance=1.0)

# Simplex class with transformations (distance <= 0 auto-calibrates eigenvalues)
s = crystal.Simplex(input_dims=100, output_dims=101, distance=-1)
s.rotate(rotation_matrix).move(offset)  # Chainable

# Rotation matrix for N dimensions (Givens rotations)
R = crystal.create_rotation_matrix(angles_matrix)

# Probability density using Wigner semicircle distribution
pdf_values = crystal.pdf(simplex, points, crystal.wigner_semicircle_distribution_pdf)
```

## Key Concepts

- **Simplex**: Regular N-dimensional geometric structure where all vertices are equidistant
- **Auto-calibration**: When `distance <= 0`, uses exact formula `d = sqrt(2N² / (N+1))` so max eigenvalue of `(MM^T / N)` ≈ 1.0
- **Wigner semicircle**: Distribution from random matrix theory: `p(x) = (2/πr²) * sqrt(r² - x²)`
- **Givens rotations**: N-dimensional rotation via composition of 2D planar rotations

## Architecture

- **Functional core**: `create_simplex_matrix()`, `create_rotation_matrix()` are pure functions
- **OOP shell**: `Simplex` class wraps state and provides chainable transformations
- **Lazy evaluation**: `simplex.matrix` property computes `offset + R @ simplex^T`

## Mathematical Notes

### Simplex Construction
1. Start with N identity basis points + one point at `((1 + sqrt(N+1))/N, ...)`
2. Center to origin by subtracting mean
3. Scale to achieve target pairwise distance

### Eigenvalue Calibration
For max eigenvalue λ = 1 in scaled Gram matrix `(MM^T / N)`:
```
distance = sqrt((2 * N²) / (N + 1))
```

### Rotation Matrix Complexity
Optimized implementation: O(N³) for N dimensions. Each Givens rotation is applied directly to rows (O(N) per rotation) instead of full matrix multiplication (O(N³) per rotation).

## Testing

Tests cover dimensions from 1D to 1000D:
- `test_simplex.py`: Simplex centering and distance properties
- `test_simplex_class.py`: Simplex class and eigenvalue calibration (λ_max ≈ 1.0 ± 0.1)
- `test_rotations.py`: Rotation orthogonality (R @ R^T = I) and distance preservation
- `test_distributions.py`: Wigner semicircle PDF calculations

## Dependencies

- numpy>=1.22.0
- pytest>=8.0.0 for testing
- setuptools>=70.0.0

## Code Style

- Uses `np.float64` instead of deprecated `np.float`
- Type hints on public API functions
- Chainable method pattern for transformations
