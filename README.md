# Quantum Orbital Simulator (Li²⁺)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Accelerated_by-Numba-FF6F00.svg)](https://numba.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance 3D visualizer and physics engine for hydrogenic electron probability densities. Built with Python, Matplotlib, SciPy, and Numba JIT acceleration, this tool solves and renders exact analytical wavefunctions for single-electron ions ($\text{Li}^{2+}$, $Z=3$) using Monte Carlo rejection sampling.

## Preview
<!-- Add your screenshot/GIF link below -->
<img width="1917" height="1022" alt="Screenshot 2026-09-03 231408" src="https://github.com/user-attachments/assets/1fcf9f61-bd17-46cc-b1e0-1a0238d8ab23" />


## Key Features
* **Accelerated Physics Engine:** JIT-compiled Monte Carlo rejection sampling powered by Numba (`numba.jit(parallel=True, fastmath=True)`) for high-speed 3D point-cloud generation.
* **Interactive 3D Visualizer:** Dynamic, auto-rotating stippled cloud representation of electron probability density ($|\psi|^2$).
* **Analytical Views:**
  * **3D Cloud:** Real-time electron density point cloud with density threshold filtering and color mapping.
  * **$P(r)$ Radial Distribution:** Plots radial probability density $r^2 |R_{n,l}(r)|^2$ alongside expectation radius $\langle r \rangle$ and most probable radius $r_{mp}$.
  * **$R(r)$ Wavefunction:** Displays radial wavefunction profiles highlighting radial nodes.
  * **Energy Spectrum:** Visualizes hydrogenic energy levels ($E_n$) with active level highlighting.
* **Real-time Parameter Manipulation:** Dynamically adjust quantum numbers ($n, l, m$), point densities, opacities, cutoffs, and camera rotation on the fly via hotkeys.

## How to Run
### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Quantum-Orbital-Simulator.git](https://github.com/your-username/Quantum-Orbital-Simulator.git)
cd Quantum-Orbital-Simulator
```
### 2. Install Dependencies
Make sure you have Python 3.8+ installed, then install all required libraries:
```bash
pip install numpy matplotlib scipy numba
```
### 3. Run the Simulation
Launch the simulation by running the main script:
```bash
python atomsimulation.py
```

## Controls

| Key | Action |
| :--- | :--- |
| `1` / `2` / `3` / `4` | Toggle Views (`1`: 3D Cloud, `2`: Radial $P(r)$, `3`: Wavefunction $R(r)$, `4`: Energy Spectrum) |
| `n` / `N` | Increase / Decrease Principal Quantum Number $n$ ($1 \le n \le 5$) |
| `l` / `L` | Increase / Decrease Azimuthal Quantum Number $l$ |
| `m` / `M` | Increase / Decrease Magnetic Quantum Number $m$ |
| `p` / `P` | Increase / Decrease Monte Carlo Point Count |
| `s` / `S` | Increase / Decrease Rendered Point Size |
| `t` / `T` | Increase / Decrease Density Cutoff Threshold |
| `a` / `A` | Increase / Decrease Point Opacity |
| `+` / `-` | Increase / Decrease Camera Rotation Speed |
| `SPACE` | Pause / Resume Camera Rotation |
| `c` | Reset Camera View |
| `r` | Reset Default Simulation Parameters |

## Physics & Mathematical Foundations

The simulator solves the time-independent Schrödinger equation for hydrogenic systems in spherical coordinates $(r, \theta, \phi)$:

$$\Psi_{n,l,m}(r, \theta, \phi) = R_{n,l}(r) Y_{l}^{m}(\theta, \phi)$$

* **Radial Component $R_{n,l}(r)$:** Evaluated using Generalized Laguerre polynomials $L_{n-l-1}^{2l+1}(\rho)$ where $\rho = \frac{2Zr}{n a_0}$.
* **Angular Component $Y_{l}^{m}(\theta, \phi)$:** Evaluated using Spherical Harmonics $Y_l^m(\theta, \phi)$ derived from Associated Legendre polynomials $P_l^m(\cos\theta)$.
* **Probability Density:** Evaluated across sampled coordinates via $P(r, \theta, \phi) = |\Psi_{n,l,m}(r, \theta, \phi)|^2$.

## Libraries Used

* **NumPy**
* **Numba**
* **SciPy**
* **Matplotlib**
