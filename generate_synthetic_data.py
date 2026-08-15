"""
generate_synthetic_data.py
==========================
Synthetic data used for Figure 1 in:
  "Enhanced SVD for Signal-Noise Separation in Spectroscopic Data"
  N. Spegazzini - Applied Spectroscopy (Note)
"""

import numpy as np


def make_synthetic(seed=42, nv=200, nt=40, snr=50, sigma=20, amp=0.25):
    """
    Parameters
    ----------
    seed  : int   - random seed (default 42)
    nv    : int   - number of spectral channels (default 200)
    nt    : int   - number of time points (default 40)
    snr   : float - signal-to-noise ratio (default 50)
    sigma : float - Gaussian band half-width in cm-1 (default 20)
    amp   : float - baseline amplitude (default 0.25)

    Returns
    -------
    D  : ndarray (nv x nt) - data matrix with baseline + noise
    wn : ndarray (nv,)     - wavenumber axis (cm-1)
    t  : ndarray (nt,)     - time axis (arb. units)
    """
    rng = np.random.default_rng(seed)

    # -- Axes -----------------------------------------------------------------
    wn = np.linspace(900, 1800, nv)   # 900-1800 cm-1
    t = np.linspace(0, 1, nt)         # 0-1 arb. units

    # -- Pure component spectra (Gaussian bands) ------------------------------
    s1 = np.exp(-((wn - 1200) / sigma) ** 2)   # reactant, 1200 cm-1
    s2 = np.exp(-((wn - 1650) / sigma) ** 2)   # product,  1650 cm-1

    # -- Kinetic profiles (first order, tau = 0.3) ----------------------------
    tau = 0.3
    c1 = np.exp(-t / tau)            # decay
    c2 = 1 - np.exp(-t / tau)        # growth

    D_true = np.outer(s1, c1) + np.outer(s2, c2)

    # -- Baseline: sinusoidal fringe whose phase drifts linearly in time ------
    # Not separable in wavenumber and time, hence not of rank one.
    x = (wn - wn[0]) / (wn[-1] - wn[0])
    baseline = amp * np.sin(2 * np.pi * (x[:, None] + 0.5 * t[None, :]))

    # -- Noise: white Gaussian, std = max(D_true + baseline) / snr ------------
    D = D_true + baseline
    D = D + rng.normal(0, np.max(D) / snr, D.shape)

    return D, wn, t


if __name__ == '__main__':
    D, wn, t = make_synthetic()
    print(f"D shape : {D.shape}")
    print(f"wn range: {wn[0]:.1f} - {wn[-1]:.1f} cm-1  (n_v={len(wn)})")
    print(f"t range : {t[0]:.2f} - {t[-1]:.2f} arb. units  (n_t={len(t)})")
    print(f"D range : {D.min():.4f} - {D.max():.4f}")
