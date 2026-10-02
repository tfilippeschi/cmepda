# MC simulation of cosmic rays detection with 3 scintillators
# Zenith angle distribution proportional to cos^2.5(theta)
# Azimuthal angle distribution uniform in [0, 2*pi]
# Length in meters
# Axis: up vertical +z, right +x

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

class CosmicRay:
    def __init__(self, exponent = 2.5, size_x = 0.40, size_y = 0.30):
        self.exponent = exponent
        self.size_x = size_x
        self.size_y = size_y
    
    def throw(self):
        cos_theta = np.random.power(self.exponent + 1.0, N)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0.0, 2.0 * np.pi, N)
        x = np.random.uniform(0.0, self.size_x, N)
        y = np.random.uniform(0.0, self.size_y, N)
        return cos_theta, theta, phi, x, y

class Detector:
    def __init__(self, xmin, xmax, ymin, ymax, z):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.z = z
    
    def CheckIntersect(self, theta, phi, x, y):
        tan_theta = np.tan(theta)
        x_on_plane = x - self.z * tan_theta * np.cos(phi)
        y_on_plane = y - self.z * tan_theta * np.sin(phi)
        inside = (
            (x_on_plane > self.xmin) &
            (x_on_plane < self.xmax) &
            (y_on_plane > self.ymin) &
            (y_on_plane < self.ymax)
        )
        return inside, x_on_plane, y_on_plane

def CosmicMC(N, exponent, size_x = 0.40, size_y = 0.30):
    ray = CosmicRay(exponent, size_x, size_y)
    cos_theta, theta, phi, x, y = ray.throw()

    down = Detector(0., size_x, 0., size_y, -0.10)
    up = Detector(0., size_x, 0., size_y, 0.10)

    hits_down, _, _ = down.CheckIntersect(theta, phi, x, y)
    hits_up, _, _ = up.CheckIntersect(theta, phi, x, y)

    pairs = np.count_nonzero(hits_down)
    triples = np.count_nonzero(hits_down & hits_up)
    
    ratio = triples / pairs if pairs > 0 else 0
    ratio_err = np.sqrt(ratio * (1.0 - ratio) / pairs) if pairs > 0 else 0.0

    def theoretical_distribution(mu, exponent):
        return (exponent + 1) * mu ** exponent
    
    n_bins = 50
    counts, bin_edges = np.histogram(cos_theta, bins=n_bins, range=(0, 1))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    y_dens = counts / (N * bin_width)
    y_err = np.sqrt(np.where(counts > 0, counts, 1)) / (N * bin_width)

    
    popt, pcov = curve_fit(
        theoretical_distribution,
        bin_centers,
        y_dens,
        p0=[exponent],
        sigma=y_err,
        absolute_sigma=True
    )

    fit_exp = popt[0]
    fit_exp_err = np.sqrt(pcov[0, 0])

    y_pred = theoretical_distribution(bin_centers, fit_exp)
    residuals = (y_dens - y_pred) / y_err
    chi2 = np.sum(residuals ** 2)
    ndof = len(bin_centers) - len(popt)

    print(f"Number of cosmic rays thrown: {N}")
    print(f"Number of pairs: {pairs}")
    print(f"Number of triples: {triples}")
    print(f"Triple / pairs = {triples} / {pairs} = {ratio:.4f} ± {ratio_err:.4f}")
    print(f"Best fit exponent: {fit_exp:.4f} ± {fit_exp_err:.4f}")
    print(f"Chi-squared: {chi2:.4f}")
    print(f"Chi-squared per degree of freedom: {chi2 / ndof:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(cos_theta, bins=50, density=True, alpha=0.7, color='blue', label='Simulated')
    mu_grid = np.linspace(0, 1, 100)
    axes[0].plot(mu_grid, theoretical_distribution(mu_grid, popt[0]), 'r-', lw=2, label=f'Fit: cos^{popt[0]:.1f}(theta)')
    axes[0].set_xlabel(r'$\cos\theta$')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_title('Zenith Angle Distribution')
    axes[0].legend(['Simulated', f'Fit: cos^{popt[0]:.1f}(theta)'])

    axes[1].hist(phi, bins=50, density=True, alpha=0.7, color='green', label='Simulated')
    axes[1].set_xlabel(r'$\phi$ [rad]')
    axes[1].set_ylabel('Probability Density')
    axes[1].set_title('Azimuthal Angle Distribution')
    axes[1].legend(['Simulated'])

    axes[2].hist2d(x[hits_down], y[hits_down], bins=50, density=True, cmap='Blues', alpha=0.7, label='Simulated', cmin=1)
    axes[2].set_xlabel('x [m]')
    axes[2].set_ylabel('y [m]')
    axes[2].set_title('Hits on Down Detector')
    axes[2].set_xlim(-0.05, 0.45)
    axes[2].set_ylim(-0.05, 0.35)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N = 100000
    exponent = 2.5
    size_x = 0.40
    size_y = 0.30
    CosmicMC(N, exponent, size_x, size_y)