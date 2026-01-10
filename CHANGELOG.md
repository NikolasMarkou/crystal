# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-01-10

### Changed
- **Eigenvalue auto-calibration**: Replaced polynomial approximation with exact analytic formula `d = sqrt(2N² / (N+1))` for computing simplex distance when max eigenvalue should be ~1.0 (`11a87d3`)
- **Rotation matrix optimization**: Reduced complexity from O(N^5) to O(N³) by applying Givens rotations directly to rows instead of full matrix multiplication (`11a87d3`)
- **NumPy compatibility**: Updated deprecated `np.float` to `np.float64` and `np.int` to `int` for modern NumPy support (`11a87d3`)

### Security
- Updated `numpy` from ~=1.19.1 to >=1.22.0 (fixes buffer overflow CVEs) (`f7ce237`)
- Updated `setuptools` from ~=50.2.0 to >=70.0.0 (fixes CVE-2022-40897 ReDoS) (`f7ce237`)
- Updated `pytest` from ~=6.0.1 to >=8.0.0 (`f7ce237`)

### Added
- `CLAUDE.md` - Project overview and development notes (`11a87d3`)
- Mathematical documentation in README.md for eigenvalue formula (`11a87d3`)

### Removed
- `test_keras.ipynb` - Removed notebook with outdated dependencies (`a3cad5d`)

## [0.1.0] - 2020-09-14

### Added
- Initial release
- `create_simplex_matrix()` - Create N-dimensional simplex structures
- `Simplex` class - Encapsulated simplex with rotation and translation
- `create_rotation_matrix()` - N-dimensional rotation matrices via Givens rotations
- `wigner_semicircle_distribution_pdf()` - Wigner semicircle distribution
- `pdf()` - Probability density function calculator
- Test suite covering 1D to 1000D dimensions
