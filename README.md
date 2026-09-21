# Quantum Orbital Simulator
This program offers a 3D visualization of electron wave-functions with the use of Numba JIT acceleration (`numba.jit(parallel=True, fastmath=True)`) and the Monte Carlo method. 

## Preview
<img width="1917" height="1022" alt="Screenshot 2026-09-03 231408" src="https://github.com/user-attachments/assets/1fcf9f61-bd17-46cc-b1e0-1a0238d8ab23" />

# Features
* Face-to-face visualization of electron distribution in real-time
* Analytical graphs for radial distribution $P(r)$, wavefunction $R(r)$, and energy spectrum
* Dynamic JIT mathematical tool with the possibility to change density and opacity of points

# Quick Start
```bash
git clone https://github.com/your-username/Quantum-Orbital-Simulator.git
```
```bash
cd Quantum-Orbital-Simulator
```
```bash
pip install numpy matplotlib scipy numba
```
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

# Libraries
NumPy | Numba | SciPy | Matplotlib

# Physics & Math
* Schrödinger Equation: 
$\Psi_{n,l,m}(r, \theta, \phi) = R_{n,l}(r) Y_{l}^{m}(\theta, \phi)$ 
* Radial Part $R_{n,l}(r)$: Solved via Generalized Laguerre polynomials $L_{n-l-1}^{2l+1}(\rho)$ where $\rho = \frac{2Zr}{n a_0}$
* Angular Part $Y_{l}^{m}(\theta, \phi)$: Solved via Spherical Harmonics $Y_l^m$ & Associated Legendre polynomials $P_l^m(\cos\theta)$
* Density Sampling: 3D point generation sampled according to probability distribution $P = \vert{}\Psi\vert{}^2$

```bash
P.S this program takes time to run please wait for 30-40 seconds for it to run
```
