import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import genlaguerre, factorial
from scipy.ndimage import gaussian_filter
from numba import jit, prange

# Scipy compatibility
try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y
    def sph_harm(m, l, phi, theta):
        return sph_harm_y(l, m, theta, phi)

# dark theme
plt.style.use('dark_background')


# NUMBA FUNCTIONS

@jit(nopython=True, fastmath=True)
def _numba_factorial(n: int) -> float:
    """factorial"""
    val = 1.0
    for i in range(2, n + 1):
        val *= i
    return val


@jit(nopython=True, fastmath=True)
def _numba_genlaguerre_eval(p: int, k: int, x: float) -> float:
    """Laguerre polynomial"""
    if p == 0:
        return 1.0
    elif p == 1:
        return 1.0 + float(k) - x
    
    l_prev2 = 1.0
    l_prev1 = 1.0 + float(k) - x
    l_curr = 0.0
    
    for i in range(2, p + 1):
        l_curr = ((2.0 * i - 1.0 + float(k) - x) * l_prev1 - (i - 1.0 + float(k)) * l_prev2) / float(i)
        l_prev2 = l_prev1
        l_prev1 = l_curr
        
    return l_curr


@jit(nopython=True, fastmath=True)
def _numba_associated_legendre_eval(l: int, m: int, x: float) -> float:
    """Legendre polynomial"""
    m_abs = abs(m)
    pmm = 1.0
    
    if m_abs > 0:
        somx2 = np.sqrt(max(0.0, (1.0 - x) * (1.0 + x)))
        fact = 1.0
        for i in range(1, m_abs + 1):
            pmm *= -fact * somx2
            fact += 2.0
            
    if l == m_abs:
        return pmm
        
    pmmp1 = x * (2.0 * m_abs + 1.0) * pmm
    if l == m_abs + 1:
        return pmmp1
        
    pll = 0.0
    for ll in range(m_abs + 2, l + 1):
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m_abs - 1.0) * pmm) / (ll - m_abs)
        pmm = pmmp1
        pmmp1 = pll
        
    return pll


@jit(nopython=True, fastmath=True)
def _numba_hydrogenic_density_point(r: float, theta: float, phi: float, 
                                    n: int, l: int, m: int, Z: float) -> float:
    """
    probability density
    """
    if r < 1e-12:
        r = 1e-12

    rho = (2.0 * Z * r) / float(n)
    p = n - l - 1
    k = 2 * l + 1
    
    fact_num = _numba_factorial(n - l - 1)
    fact_den = _numba_factorial(n + l)
    norm_r = np.sqrt(((2.0 * Z / float(n)) ** 3) * fact_num / (2.0 * float(n) * ((fact_den) ** 3)))
    
    laguerre_val = _numba_genlaguerre_eval(p, k, rho)
    R_nl = norm_r * np.exp(-rho / 2.0) * (rho ** l) * laguerre_val
    
    m_abs = abs(m)
    norm_y_num = (2.0 * float(l) + 1.0) * _numba_factorial(l - m_abs)
    norm_y_den = 4.0 * np.pi * _numba_factorial(l + m_abs)
    norm_y = np.sqrt(norm_y_num / norm_y_den)
    
    cos_theta = np.cos(theta)
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
        
    legendre_val = _numba_associated_legendre_eval(l, m_abs, cos_theta)
    Y_lm_abs = norm_y * abs(legendre_val)
    
    psi_mag = abs(R_nl * Y_lm_abs)
    return psi_mag * psi_mag


@jit(nopython=True, parallel=True, fastmath=True)
def monte_carlo_rejection_sampling(n: int, l: int, m: int, Z: float, 
                                   num_target_points: int, r_max: float, 
                                   p_max: float) -> tuple:
    """
    Monte Carlo sampling 
    """
    x_out = np.empty(num_target_points, dtype=np.float64)
    y_out = np.empty(num_target_points, dtype=np.float64)
    z_out = np.empty(num_target_points, dtype=np.float64)
    prob_out = np.empty(num_target_points, dtype=np.float64)
    
    chunk_size = 10000
    collected = 0
    
    while collected < num_target_points:
        x_rand = (2.0 * np.random.random(chunk_size) - 1.0) * r_max
        y_rand = (2.0 * np.random.random(chunk_size) - 1.0) * r_max
        z_rand = (2.0 * np.random.random(chunk_size) - 1.0) * r_max
        u_rand = np.random.random(chunk_size) * p_max
        
        for idx in range(chunk_size):
            rx = x_rand[idx]
            ry = y_rand[idx]
            rz = z_rand[idx]
            r_sq = rx*rx + ry*ry + rz*rz
            r = np.sqrt(r_sq)
            
            if r > r_max:
                continue
                
            theta = np.arccos(rz / r) if r > 1e-12 else 0.0
            phi = np.arctan2(ry, rx)
            if phi < 0.0:
                phi += 2.0 * np.pi
                
            density = _numba_hydrogenic_density_point(r, theta, phi, n, l, m, Z)
            
            if u_rand[idx] <= density:
                if collected < num_target_points:
                    x_out[collected] = rx
                    y_out[collected] = ry
                    z_out[collected] = rz
                    prob_out[collected] = density
                    collected += 1
                    
    return x_out, y_out, z_out, prob_out


# ORBITAL MATHS

class HydrogenicOrbital:
    """Hydrogenic orbital"""

    def __init__(self, n: int = 3, l: int = 2, m: int = 1, Z: float = 3.0):
        self.Z = Z
        self.a0 = 1.0  # Bohr radius
        self.set_quantum_numbers(n, l, m)

    def set_quantum_numbers(self, n: int, l: int, m: int):
        if not (1 <= n <= 5):
            raise ValueError(f"Invalid n={n}. Principal quantum number must be 1 <= n <= 5.")
        if not (0 <= l < n):
            raise ValueError(f"Invalid l={l} for n={n}. Azimuthal quantum number must be 0 <= l < n.")
        if not (-l <= m <= l):
            raise ValueError(f"Invalid m={m} for l={l}. Magnetic quantum number must be -l <= m <= l.")

        self.n = int(n)
        self.l = int(l)
        self.m = int(m)

    @property
    def orbital_notation(self) -> str:
        symbols = ['s', 'p', 'd', 'f', 'g']
        sym = symbols[self.l] if self.l < len(symbols) else f'l={self.l}'
        return f"{self.n}{sym}"

    def get_energy_ev(self) -> float:
        """energy level"""
        return -13.60569 * (self.Z ** 2) / (self.n ** 2)

    def get_characteristic_radius(self) -> float:
        """characteristic radius"""
        r_avg = self.get_expectation_r()
        return max(4.0, 3.6 * r_avg)

    def get_radial_nodes(self) -> int:
        return self.n - self.l - 1

    def get_angular_nodes(self) -> int:
        return self.l

    def get_total_nodes(self) -> int:
        return self.n - 1

    def get_expectation_r(self) -> float:
        """expectation radius"""
        return (self.a0 / (2.0 * self.Z)) * (3.0 * (self.n ** 2) - self.l * (self.l + 1))

    def compute_radial_component(self, r: np.ndarray) -> np.ndarray:
        """radial wavefunction"""
        rho = (2.0 * self.Z * r) / (self.n * self.a0)
        p = self.n - self.l - 1
        k = 2 * self.l + 1

        laguerre_poly = genlaguerre(p, k)
        laguerre_vals = laguerre_poly(rho)

        fact_num = float(factorial(self.n - self.l - 1))
        fact_den = float(factorial(self.n + self.l))
        norm_r = np.sqrt(((2.0 * self.Z / (self.n * self.a0)) ** 3) * fact_num / (2.0 * self.n * (fact_den ** 3)))

        return norm_r * np.exp(-rho / 2.0) * (rho ** self.l) * laguerre_vals

    def compute_angular_component(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """spherical harmonic"""
        return sph_harm(self.m, self.l, phi, theta)

    def evaluate_wavefunction(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.complex128:
        """ wavefunction """
        R_nl = self.compute_radial_component(r)
        Y_lm = self.compute_angular_component(theta, phi)
        return R_nl * Y_lm

    def evaluate_probability_density(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """probability density"""
        psi = self.evaluate_wavefunction(r, theta, phi)
        return np.abs(psi) ** 2

    def compute_radial_probability_distribution(self, r_pts: np.ndarray) -> tuple:
        """ radial probability distribution """
        R_nl = self.compute_radial_component(r_pts)
        P_r = (r_pts ** 2) * (R_nl ** 2)
        return R_nl, P_r

    def get_most_probable_radius(self) -> float:
        """ most probable radius """
        r_max_limit = self.get_characteristic_radius() * 1.5
        r_dense = np.linspace(0.0, r_max_limit, 10000)
        _, P_r = self.compute_radial_probability_distribution(r_dense)
        return float(r_dense[np.argmax(P_r)])

    def estimate_peak_density(self) -> float:
        """ peak density """
        r_max = self.get_characteristic_radius()
        r_pts = np.linspace(0.001, r_max, 150)
        theta_pts = np.linspace(0, np.pi, 60)
        phi_pts = np.linspace(0, 2 * np.pi, 60)

        R, T, P = np.meshgrid(r_pts, theta_pts, phi_pts, indexing='ij')
        density_grid = self.evaluate_probability_density(R, T, P)
        return float(np.max(density_grid)) * 1.15

    def verify_normalization_integral(self) -> float:
        """Normalization check"""
        r_max_integration = self.get_characteristic_radius() * 4.0
        r_pts, dr = np.linspace(0.0, r_max_integration, 30000, retstep=True)
        _, P_r = self.compute_radial_probability_distribution(r_pts)
        return float(np.trapezoid(P_r, r_pts))


# GRID PROCESSING

def generate_cartesian_spherical_grid(r_max: float, grid_res: int):
    x = np.linspace(-r_max, r_max, grid_res)
    y = np.linspace(-r_max, r_max, grid_res)
    z = np.linspace(-r_max, r_max, grid_res)
    dx = x[1] - x[0]
    dV = dx ** 3

    X, Y, Z_grid = np.meshgrid(x, y, z, indexing='ij')

    R = np.sqrt(X**2 + Y**2 + Z_grid**2)
    R_safe = np.where(R == 0, 1e-12, R)

    Theta = np.arccos(np.clip(Z_grid / R_safe, -1.0, 1.0))
    Phi = np.arctan2(Y, X)
    Phi = np.where(Phi < 0, Phi + 2 * np.pi, Phi)

    return x, y, z, X, Y, Z_grid, R, Theta, Phi, dV


def compute_smoothed_volume_field(orbital: HydrogenicOrbital, grid_res: int = 60, 
                                  sigma: float = 0.8) -> tuple:
    r_max = orbital.get_characteristic_radius()
    x, y, z, X, Y, Z_grid, R, Theta, Phi, dV = generate_cartesian_spherical_grid(r_max, grid_res)

    density_field = orbital.evaluate_probability_density(R, Theta, Phi)
    total_prob = float(np.sum(density_field) * dV)

    if total_prob > 0:
        density_field = density_field / total_prob

    density_smoothed = gaussian_filter(density_field, sigma=sigma)
    return X, Y, Z_grid, density_smoothed, dV, total_prob


# SIMULATOR

class QuantumOrbitalSimulator:
    """Quantum Orbital Viewer"""

    def __init__(self):
        # orbital settings
        self.n = 3
        self.l = 2
        self.m = 1
        self.Z = 3.0
        self.orbital = HydrogenicOrbital(self.n, self.l, self.m, self.Z)

        # display settings
        self.num_points = 35000
        self.point_size = 10.0
        self.alpha = 0.70
        self.density_threshold_pct = 2.0
        self.custom_cmap = 'inferno'

        # Camera 
        self.azimuth_angle = 45.0
        self.elevation_angle = 20.0
        self.rotation_speed = 0.8
        self.animation_interval = 30
        self.is_rotating = True

        # View Mode 
        self.active_view = 'cloud'

        # point data
        self.px = np.array([])
        self.py = np.array([])
        self.pz = np.array([])
        self.p_density = np.array([])

        # plot handles
        self.colorbar = None
        self.scatter_handle = None

        # window setup 
        self._setup_figure()

        # intial render
        self.recompute_physics()

        # keyboard input
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def _setup_figure(self):
        """Set up the figure."""
        self.fig = plt.figure(figsize=(16, 9), facecolor='black')
        self.fig.canvas.manager.set_window_title("Li²⁺ Quantum Mechanical Simulator")

        # Main Axes
        self.ax_main = self.fig.add_subplot(111, facecolor='black')
        self.ax_main.set_position([0.05, 0.08, 0.65, 0.84])

        # 3S axes
        self.ax3d = self.fig.add_subplot(111, projection='3d', facecolor='black')
        self.ax3d.set_position([0.05, 0.08, 0.65, 0.84])

        # Colorbar 
        self.cax = self.fig.add_axes([0.72, 0.20, 0.018, 0.60])

        # Info panel
        self.info_panel_handle = self.fig.text(
            0.76, 0.95, "", color='white', fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#111116', edgecolor='#333344', alpha=0.90)
        )

        # Control Panels
        self.controls_hud_handle = self.fig.text(
            0.76, 0.38,
            "KEYBOARD CONTROLS\n"
            "-----------------\n"
            "n / N : Increase / Decrease n (1..5)\n"
            "l / L : Increase / Decrease l\n"
            "m / M : Increase / Decrease m\n"
            "p / P : Increase / Decrease Point Count\n"
            "s / S : Increase / Decrease Point Size\n"
            "t / T : Increase / Decrease Density Cutoff\n"
            "a / A : Increase / Decrease Opacity\n"
            "1..4  : View Toggles (1:Cloud, 2:P(r), 3:R(r), 4:Energy)\n"
            "+ / - : Faster / Slower Rotation\n"
            "SPACE : Pause / Resume Camera Rotation\n"
            "c / C : Reset Camera Orientation\n"
            "r / R : Reset Default Parameters\n",
            color='#CCCCCC', fontsize=7.5,
            verticalalignment='top', fontfamily='monospace'
        )

    def recompute_physics(self):
        
        self.orbital.set_quantum_numbers(self.n, self.l, self.m)
        r_max = self.orbital.get_characteristic_radius()
        p_max = self.orbital.estimate_peak_density()

        raw_x, raw_y, raw_z, raw_prob = monte_carlo_rejection_sampling(
            self.n, self.l, self.m, self.Z, self.num_points, r_max, p_max
        )

        if len(raw_prob) > 0:
            threshold_val = np.percentile(raw_prob, self.density_threshold_pct)
            mask = raw_prob >= threshold_val
            self.px = raw_x[mask]
            self.py = raw_y[mask]
            self.pz = raw_z[mask]
            self.p_density = raw_prob[mask]
        else:
            self.px, self.py, self.pz, self.p_density = raw_x, raw_y, raw_z, raw_prob

        self._update_info_panel()
        self.render_active_view()

    def _update_info_panel(self):
        
        E_ev = self.orbital.get_energy_ev()
        r_avg = self.orbital.get_expectation_r()
        r_mp = self.orbital.get_most_probable_radius()
        r_nodes = self.orbital.get_radial_nodes()
        a_nodes = self.orbital.get_angular_nodes()
        norm_val = self.orbital.verify_normalization_integral()

        info_str = (
            f"SYSTEM & QUANTUM NUMBERS\n"
            f"------------------------\n"
            f"Ion Species       : Li²⁺ (Hydrogenic)\n"
            f"Atomic Number (Z) : 3  (Nuclear Charge: +3e)\n"
            f"Electron Count    : 1\n"
            f"Orbital Notation  : {self.orbital.orbital_notation}\n"
            f"Principal (n)     : {self.n}\n"
            f"Azimuthal (l)     : {self.l}\n"
            f"Magnetic (m)      : {self.m}\n\n"
            f"PHYSICAL OBSERVABLES\n"
            f"--------------------\n"
            f"Energy Level (E_n): {E_ev:.3f} eV\n"
            f"Expectation <r>   : {r_avg:.3f} a₀/Z\n"
            f"Most Probable r   : {r_mp:.3f} a₀/Z\n"
            f"Radial Nodes      : {r_nodes}\n"
            f"Angular Nodes     : {a_nodes}\n"
            f"Total Nodes       : {r_nodes + a_nodes}\n"
            f"Integral Norm ∫P  : {norm_val:.5f}"
        )
        self.info_panel_handle.set_text(info_str)

    def render_active_view(self):
        
        if self.active_view == 'cloud':
            self.ax_main.set_visible(False)
            self.ax3d.set_visible(True)
            self.cax.set_visible(True)
            self._render_stippled_cloud()
        else:
            self.ax3d.set_visible(False)
            self.cax.set_visible(False)
            self.ax_main.set_visible(True)
            self.ax_main.clear()

            if self.active_view == 'radial_prob':
                self._render_radial_probability_view()
            elif self.active_view == 'wavefunction':
                self._render_wavefunction_view()
            elif self.active_view == 'energy':
                self._render_energy_diagram_view()

        self.fig.canvas.draw_idle()

    def _render_stippled_cloud(self):
        
        self.ax3d.clear()
        self.cax.clear()

        # Dark theme axes styling
        self.ax3d.xaxis.pane.fill = False
        self.ax3d.yaxis.pane.fill = False
        self.ax3d.zaxis.pane.fill = False
        

        self.ax3d.xaxis._axinfo['grid']['color'] = (0.30, 0.30, 0.38, 0.50)
        self.ax3d.yaxis._axinfo['grid']['color'] = (0.30, 0.30, 0.38, 0.50)
        self.ax3d.zaxis._axinfo['grid']['color'] = (0.30, 0.30, 0.38, 0.50)

        self.ax3d.tick_params(colors='white', labelsize=8)
        self.ax3d.set_xlabel('X (a₀/Z)', color='white', labelpad=8, fontsize=9)
        self.ax3d.set_ylabel('Y (a₀/Z)', color='white', labelpad=8, fontsize=9)
        self.ax3d.set_zlabel('Z (a₀/Z)', color='white', labelpad=8, fontsize=9)

        if len(self.px) > 0:
            p_min = np.min(self.p_density)
            p_max = np.max(self.p_density)

            self.scatter_handle = self.ax3d.scatter(
                self.px, self.py, self.pz,
                c=self.p_density,
                cmap=self.custom_cmap,
                vmin=p_min,
                vmax=p_max,
                s=self.point_size,
                alpha=self.alpha,
                edgecolors='none'
            )

            self.colorbar = self.fig.colorbar(
                self.scatter_handle,
                cax=self.cax,
                orientation='vertical'
            )
            self.colorbar.set_label('Probability Density |ψ|²', color='white', fontsize=10, labelpad=10)
            self.colorbar.ax.tick_params(colors='white', labelsize=8)

        r_max = self.orbital.get_characteristic_radius()
        self.ax3d.set_xlim([-r_max, r_max])
        self.ax3d.set_ylim([-r_max, r_max])
        self.ax3d.set_zlim([-r_max, r_max])
        self.ax3d.set_box_aspect([1, 1, 1])

        self.ax3d.set_title(
            f"Stippled 3D Electron Density Cloud: Li²⁺ ({self.orbital.orbital_notation})\n"
            f"Points: {len(self.px):,} | Threshold Cutoff: {self.density_threshold_pct:.1f}%",
            color='white', fontsize=11, pad=12
        )

    def _render_radial_probability_view(self):
        """Plot radial probability distribution """
        r_max = self.orbital.get_characteristic_radius() * 1.3
        r_pts = np.linspace(0, r_max, 1000)
        _, P_r = self.orbital.compute_radial_probability_distribution(r_pts)

        self.ax_main.set_facecolor('#0A0A0E')
        self.ax_main.plot(r_pts, P_r, color='#FF6D00', linewidth=2.5, label='P(r) = r² |R_{n,l}(r)|²')
        self.ax_main.fill_between(r_pts, 0, P_r, color='#FF6D00', alpha=0.30)

        r_mp = self.orbital.get_most_probable_radius()
        r_avg = self.orbital.get_expectation_r()
        self.ax_main.axvline(r_mp, color='#FFD600', linestyle='--', linewidth=1.5, label=f'r_mp = {r_mp:.2f} a₀/Z')
        self.ax_main.axvline(r_avg, color='#00E676', linestyle=':', linewidth=1.5, label=f'<r> = {r_avg:.2f} a₀/Z')

        self.ax_main.set_title(f"Radial Probability Distribution P(r) [{self.orbital.orbital_notation}]", color='white', fontsize=12)
        self.ax_main.set_xlabel("Radius r (a₀/Z)", color='white', fontsize=10)
        self.ax_main.set_ylabel("Probability Density P(r)", color='white', fontsize=10)
        self.ax_main.tick_params(colors='white', labelsize=9)
        self.ax_main.grid(True, color='#222233', linestyle=':')
        self.ax_main.legend(facecolor='#111115', edgecolor='#333344', labelcolor='white', fontsize=9)

    def _render_wavefunction_view(self):
        
        r_max = self.orbital.get_characteristic_radius() * 1.3
        r_pts = np.linspace(0, r_max, 1000)
        R_nl, _ = self.orbital.compute_radial_probability_distribution(r_pts)

        self.ax_main.set_facecolor('#0A0A0E')
        self.ax_main.plot(r_pts, R_nl, color='#00E5FF', linewidth=2.5, label=f'R_{{{self.n},{self.l}}}(r)')
        self.ax_main.axhline(0, color='#666666', linestyle='--', linewidth=1.0)

        self.ax_main.set_title(f"Radial Wavefunction R(r) [{self.orbital.orbital_notation}]", color='white', fontsize=12)
        self.ax_main.set_xlabel("Radius r (a₀/Z)", color='white', fontsize=10)
        self.ax_main.set_ylabel("R_{n,l}(r)", color='white', fontsize=10)
        self.ax_main.tick_params(colors='white', labelsize=9)
        self.ax_main.grid(True, color='#222233', linestyle=':')
        self.ax_main.legend(facecolor='#111115', edgecolor='#333344', labelcolor='white', fontsize=9)

    def _render_energy_diagram_view(self):
        """Plot energy levels"""
        self.ax_main.set_facecolor('#0A0A0E')
        n_levels = np.arange(1, 6)
        energies = [-13.60569 * (self.Z ** 2) / (n_i ** 2) for n_i in n_levels]

        for idx, n_i in enumerate(n_levels):
            E_i = energies[idx]
            is_current = (n_i == self.n)
            line_color = '#FF1744' if is_current else '#7777AA'
            line_width = 3.5 if is_current else 1.8
            alpha_val = 1.0 if is_current else 0.65

            self.ax_main.hlines(E_i, 0.2, 0.8, colors=line_color, linewidth=line_width, alpha=alpha_val)
            text_label = f"n={n_i} ({E_i:.2f} eV)" + ("  ← ACTIVE LEVEL" if is_current else "")
            self.ax_main.text(0.82, E_i, text_label, color=line_color, 
                              verticalalignment='center', fontsize=9, fontweight='bold' if is_current else 'normal')

        self.ax_main.set_xlim(0, 1.3)
        self.ax_main.set_xticks([])
        self.ax_main.set_ylabel("Energy E_n (eV)", color='white', fontsize=10)
        self.ax_main.set_title(f"Hydrogenic Li²⁺ Energy Spectrum (Selected Level: n={self.n})", color='white', fontsize=12)
        self.ax_main.tick_params(colors='white', labelsize=9)
        self.ax_main.grid(True, axis='y', color='#222233', linestyle=':')

    def on_key_press(self, event):
        """Keyboard input"""
        key = event.key
        need_recompute = False
        need_redraw_only = False

        if key == 'n':
            if self.n < 5:
                self.n += 1
                if self.l >= self.n:
                    self.l = self.n - 1
                if abs(self.m) > self.l:
                    self.m = 0
                need_recompute = True

        elif key == 'N':
            if self.n > 1:
                self.n -= 1
                if self.l >= self.n:
                    self.l = self.n - 1
                if abs(self.m) > self.l:
                    self.m = 0
                need_recompute = True

        elif key == 'l':
            if self.l < self.n - 1:
                self.l += 1
                if abs(self.m) > self.l:
                    self.m = 0
                need_recompute = True

        elif key == 'L':
            if self.l > 0:
                self.l -= 1
                if abs(self.m) > self.l:
                    self.m = 0
                need_recompute = True

        elif key == 'm':
            if self.m < self.l:
                self.m += 1
                need_recompute = True

        elif key == 'M':
            if self.m > -self.l:
                self.m -= 1
                need_recompute = True

        elif key == 'p':
            self.num_points = min(150000, self.num_points + 5000)
            need_recompute = True

        elif key == 'P':
            self.num_points = max(2000, self.num_points - 5000)
            need_recompute = True

        elif key == 's':
            self.point_size = min(35.0, self.point_size + 1.0)
            need_redraw_only = True

        elif key == 'S':
            self.point_size = max(0.5, self.point_size - 1.0)
            need_redraw_only = True

        elif key == 't':
            self.density_threshold_pct = min(15.0, self.density_threshold_pct + 0.5)
            need_recompute = True

        elif key == 'T':
            self.density_threshold_pct = max(0.0, self.density_threshold_pct - 0.5)
            need_recompute = True

        elif key == 'a':
            self.alpha = min(1.0, self.alpha + 0.05)
            need_redraw_only = True

        elif key == 'A':
            self.alpha = max(0.05, self.alpha - 0.05)
            need_redraw_only = True

        elif key == '1':
            self.active_view = 'cloud'
            need_redraw_only = True

        elif key == '2':
            self.active_view = 'radial_prob'
            need_redraw_only = True

        elif key == '3':
            self.active_view = 'wavefunction'
            need_redraw_only = True

        elif key == '4':
            self.active_view = 'energy'
            need_redraw_only = True

        elif key == '+':
            self.rotation_speed = min(5.0, self.rotation_speed + 0.2)

        elif key == '-':
            self.rotation_speed = max(0.0, self.rotation_speed - 0.2)

        elif key == ' ':
            self.is_rotating = not self.is_rotating

        elif key in ['c', 'C']:
            self.azimuth_angle = 45.0
            self.elevation_angle = 20.0
            self.ax3d.view_init(elev=self.elevation_angle, azim=self.azimuth_angle)

        elif key in ['r', 'R']:
            self.n, self.l, self.m = 3, 2, 1
            self.num_points = 35000
            self.point_size = 10.0
            self.alpha = 0.70
            self.density_threshold_pct = 2.0
            self.rotation_speed = 0.8
            self.azimuth_angle = 45.0
            self.elevation_angle = 20.0
            self.active_view = 'cloud'
            need_recompute = True

        if need_recompute:
            self.recompute_physics()
        elif need_redraw_only:
            self.render_active_view()

    def update_animation_frame(self, frame_idx: int):
        """Camera animation"""
        if self.active_view == 'cloud' and self.is_rotating:
            self.azimuth_angle = (self.azimuth_angle + self.rotation_speed) % 360.0
            self.elevation_angle = 15.0 + 10.0 * np.sin(np.radians(frame_idx * 0.5))
            self.ax3d.view_init(elev=self.elevation_angle, azim=self.azimuth_angle)
        return self.scatter_handle,


# MAIN

def main():
    print("  SCIENTIFIC QUANTUM MECHANICAL ORBITAL SIMULATOR (Li²⁺, Z = 3)")
    print("Pre-compiling Numba JIT math engine... Please wait...")

    # first run
    _ = monte_carlo_rejection_sampling(1, 0, 0, 3.0, 100, 2.0, 0.5)
    print("Numba JIT Engine ready!")
    print("Launching Quantum Orbital Viewer GUI...\n")

    sim = QuantumOrbitalSimulator()

    anim = FuncAnimation(
        sim.fig,
        sim.update_animation_frame,
        interval=sim.animation_interval,
        blit=False,
        cache_frame_data=False
    )

    plt.show()


if __name__ == '__main__':
    main()
