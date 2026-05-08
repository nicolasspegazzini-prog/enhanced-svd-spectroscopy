"""
Enhanced SVD for Signal-Noise Separation in Spectroscopic Data
===============================================================

Implementation based on:
    Lórenz-Fonfría, V. A., & Kandori, H. (2009). 
    "Spectroscopic and Kinetic Evidence on How Bacteriorhodopsin 
    Accomplishes Vectorial Proton Transport under Functional Conditions"
    J. Am. Chem. Soc., 131(16), 5891-5901.

FFT derivative with apodization based on:
    Kalka, A.J. & Turek, A.M.
    "Searching for Alternatives to the Savitzky-Golay Filter
    in the Spectral Processing Domain"
    Appl. Spectrosc. 2023, 77(4), 426-432

Author: Nicolas Spegazzini
Date: January 2026
"""

import numpy as np
from scipy.linalg import svd
from scipy.fft import fft, ifft


# =============================================================================
# DERIVATIVE FUNCTIONS
# =============================================================================

def derivative_fft_apodized(spectrum, deriv_order=1, apod_start=0.10, apod_end=0.30):
    """
    Compute spectral derivative using FFT with apodization to avoid Gibbs phenomenon.
    
    Parameters
    ----------
    spectrum : array_like
        1D input spectrum
    deriv_order : int, optional
        Order of derivative (default: 1)
    apod_start : float, optional
        Normalized frequency where apodization starts (0-1, default: 0.10)
    apod_end : float, optional
        Normalized frequency where signal is forced to zero (0-1, default: 0.30)
    
    Returns
    -------
    spectrum_deriv : ndarray
        Derivative of input spectrum
    
    Notes
    -----
    The function applies:
    1. Edge extension with linear extrapolation
    2. Boundary apodization (sigmoidal)
    3. FFT derivative with frequency multiplication
    4. Low-pass filter in frequency domain
    5. Inverse FFT
    """
    y = np.asarray(spectrum, dtype=float).flatten()
    max_x = len(y)
    
    # Edge extension (8% of spectrum length)
    ex_val_threshold = 0.08
    ex_val = max(int(np.ceil(max_x * ex_val_threshold)), 15)
    
    # Linear extrapolation at edges
    n_points = 5
    p_left = np.polyfit(np.arange(n_points), y[:n_points], 1)
    y_left = np.polyval(p_left, np.arange(-ex_val, 0))
    
    p_right = np.polyfit(np.arange(-n_points + 1, 1), y[-n_points:], 1)
    y_right = np.polyval(p_right, np.arange(1, ex_val + 1))
    
    y_ex = np.concatenate([y_left, y, y_right])
    
    # Boundary apodization (sigmoidal function)
    x_apod = np.arange(1, ex_val + 1) - np.median(np.arange(1, ex_val + 1))
    apod_FWHM = 0.5 * ex_val / 2 / np.log(999)
    apod_func_edge = 1 / (1 + np.exp(-x_apod / apod_FWHM))
    apod_func_edge[apod_func_edge < 1e-4] = 0
    apod_func = np.concatenate([apod_func_edge, np.ones(len(y)), apod_func_edge[::-1]])
    
    # Apply boundary apodization and normalize
    y_apod = y_ex * apod_func
    norm_scale = np.abs(np.trapezoid(y_apod)) if hasattr(np, 'trapezoid') else np.abs(np.trapz(y_apod))
    if norm_scale > 0:
        y_apod = y_apod / norm_scale
    
    # FFT
    n_fft = 12
    fft_y = fft(y_apod, 2**n_fft)
    x_fft = np.arange(1, 2**(n_fft - 1) + 1)
    fft_y = fft_y[:len(x_fft)]
    
    # Derivative in frequency domain
    fft_dy = fft_y * (2 * np.pi * 1j * x_fft / 2**n_fft)**deriv_order
    
    # Frequency domain apodization (low-pass filter)
    fft_apod_start = apod_start * len(x_fft)
    fft_apod_end = apod_end * len(x_fft)
    fft_apod_param_x0 = (fft_apod_end + fft_apod_start) / 2
    fft_apod_param_fwhm = (fft_apod_end - fft_apod_start) / 2 / np.log(999)
    
    fft_apod_func = 1 - 1 / (1 + np.exp(-(x_fft - fft_apod_param_x0) / (fft_apod_param_fwhm + 1e-10)))
    fft_apod_func[fft_apod_func < 1e-4] = 0
    
    # Apply filter and inverse FFT
    fft_dy_apod = fft_dy * fft_apod_func
    dy_ex = ifft(fft_dy_apod, 2**n_fft).real * 2
    
    # Extract original domain
    dy = dy_ex[ex_val:ex_val + len(y)]
    
    # Sign correction for derivative order
    dy = dy * (-1)**(deriv_order + 1)
    
    # Restore scale
    dy = dy * norm_scale
    
    return dy


def derivative_matrix(D, deriv_order=1, apod_start=0.10, apod_end=0.30):
    """
    Apply apodized FFT derivative to each column of a data matrix.
    
    Parameters
    ----------
    D : ndarray
        Data matrix (nv × nt), wavenumbers × time points
    deriv_order : int, optional
        Order of derivative (default: 1)
    apod_start : float, optional
        Normalized frequency where apodization starts (default: 0.10)
    apod_end : float, optional
        Normalized frequency where signal is forced to zero (default: 0.30)
    
    Returns
    -------
    D_deriv : ndarray
        Derivative of data matrix (same shape as D)
    """
    nv, nt = D.shape
    D_deriv = np.zeros_like(D, dtype=float)
    for j in range(nt):
        D_deriv[:, j] = derivative_fft_apodized(D[:, j], deriv_order, apod_start, apod_end)
    return D_deriv


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def integrate_fft(spectrum):
    """
    Integrate spectrum using FFT (inverse of derivative).
    
    Parameters
    ----------
    spectrum : array_like
        1D input spectrum (typically a derivative)
    
    Returns
    -------
    spectrum_int : ndarray
        Integrated spectrum (mean-centered)
    """
    y = np.asarray(spectrum, dtype=float).flatten()
    N = len(y)
    
    y_fft = fft(y)
    k = np.fft.fftfreq(N) * N
    
    # Avoid division by zero at DC
    k[0] = 1
    int_fft = y_fft / (1j * 2 * np.pi * k / N)
    int_fft[0] = 0  # Set DC to zero
    
    y_int = ifft(int_fft).real
    y_int = y_int - np.mean(y_int)  # Mean-center
    
    return y_int


def integrate_matrix(V_deriv):
    """
    Integrate each column of a matrix using FFT.
    
    Parameters
    ----------
    V_deriv : ndarray
        Matrix of derivative spectra (nv × ns)
    
    Returns
    -------
    V_int : ndarray
        Matrix of integrated spectra (same shape)
    """
    nv, ns = V_deriv.shape
    V_int = np.zeros_like(V_deriv, dtype=float)
    for j in range(ns):
        V_int[:, j] = integrate_fft(V_deriv[:, j])
    return V_int


# =============================================================================
# NOISE WEIGHT ESTIMATION
# =============================================================================

def estimate_noise_weights(D, method='std'):
    """
    Estimate noise-dependent weights for wavenumber and time dimensions.
    
    Parameters
    ----------
    D : ndarray
        Data matrix (nv × nt)
    method : str, optional
        Weight estimation method: 'std' (default) or 'mad'
    
    Returns
    -------
    wv : ndarray
        Wavenumber weights (nv,), inverse of noise std
    wt : ndarray
        Time weights (nt,), inverse of noise std
    
    Notes
    -----
    Weights are normalized to have mean = 1.
    """
    if method == 'std':
        noise_v = np.std(D, axis=1)
        noise_t = np.std(D, axis=0)
    elif method == 'mad':
        # Median absolute deviation (more robust)
        noise_v = np.median(np.abs(D - np.median(D, axis=1, keepdims=True)), axis=1)
        noise_t = np.median(np.abs(D - np.median(D, axis=0, keepdims=True)), axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    wv = 1 / (noise_v + 1e-10)
    wv = wv / np.mean(wv)
    
    wt = 1 / (noise_t + 1e-10)
    wt = wt / np.mean(wt)
    
    return wv, wt


# =============================================================================
# MAIN SVD FUNCTIONS
# =============================================================================

def enhanced_svd(D, wv=None, wt=None, n_components=None, 
                 apod_start=0.10, apod_end=0.30, return_all=False):
    """
    Perform SVD with baseline fluctuation removal.
    
    This method preprocesses the data with first derivative and noise weighting
    to suppress baseline fluctuations, then recovers the original spectra and
    time-traces.
    
    Parameters
    ----------
    D : ndarray
        Data matrix (nv × nt), wavenumbers × time points
    wv : ndarray, optional
        Wavenumber weights. If None, estimated from data.
    wt : ndarray, optional
        Time weights. If None, estimated from data.
    n_components : int, optional
        Number of components to return. If None, returns min(nv, nt).
    apod_start : float, optional
        Apodization start for FFT derivative (default: 0.10)
    apod_end : float, optional
        Apodization end for FFT derivative (default: 0.30)
    return_all : bool, optional
        If True, return additional outputs (default: False)
    
    Returns
    -------
    V : ndarray
        Recovered spectra matrix (nv × n_components)
    S : ndarray
        Singular values (n_components,)
    U : ndarray
        Recovered time-traces matrix (nt × n_components)
    
    If return_all=True, also returns:
    info : dict
        Dictionary with additional information:
        - 's_all': all singular values
        - 'wv': wavenumber weights used
        - 'wt': time weights used
        - 'Vw': weighted spectra (before integration)
        - 'Uw': weighted time-traces
    
    Notes
    -----
    The algorithm follows Lórenz-Fonfría & Kandori (2009):
    1. Compute first derivative: der[D]
    2. Apply noise weighting: Dw' = diag(wv) × der[D] × diag(wt)
    3. Perform SVD: Dw' = Vw' × S × Uw'
    4. Recover U: U = Uw × diag(wt)^(-1)
    5. Recover V: V = int[diag(wv)^(-1) × Vw']
    """
    D = np.asarray(D, dtype=float)
    nv, nt = D.shape
    
    # Estimate weights if not provided
    if wv is None or wt is None:
        wv_est, wt_est = estimate_noise_weights(D)
        if wv is None:
            wv = wv_est
        if wt is None:
            wt = wt_est
    
    # Step 1: Compute derivative
    D_deriv = derivative_matrix(D, deriv_order=1, apod_start=apod_start, apod_end=apod_end)
    
    # Step 2: Apply weighting
    Dw_prime = np.diag(wv) @ D_deriv @ np.diag(wt)
    
    # Step 3: SVD
    Vw, s_all, Uwt = svd(Dw_prime, full_matrices=False)
    Uw = Uwt.T
    
    # Number of components
    if n_components is None:
        n_components = min(nv, nt)
    n_components = min(n_components, len(s_all))
    
    # Step 4 & 5: Recover U and V
    U = np.zeros((nt, n_components), dtype=float)
    V = np.zeros((nv, n_components), dtype=float)
    
    for i in range(n_components):
        # Recover U (remove time weighting)
        U[:, i] = Uw[:, i] / wt
        
        # Recover V (remove wavenumber weighting, then integrate)
        v_unweighted = Vw[:, i] / wv
        V[:, i] = integrate_fft(v_unweighted)
    
    S = s_all[:n_components]
    
    if return_all:
        info = {
            's_all': s_all,
            'wv': wv,
            'wt': wt,
            'Vw': Vw[:, :n_components],
            'Uw': Uw[:, :n_components]
        }
        return V, S, U, info
    
    return V, S, U


def standard_svd(D, n_components=None):
    """
    Perform standard SVD on data matrix.
    
    Parameters
    ----------
    D : ndarray
        Data matrix (nv × nt)
    n_components : int, optional
        Number of components to return
    
    Returns
    -------
    V : ndarray
        Spectra matrix (nv × n_components)
    S : ndarray
        Singular values (n_components,)
    U : ndarray
        Time-traces matrix (nt × n_components)
    """
    D = np.asarray(D, dtype=float)
    nv, nt = D.shape
    
    V, s_all, Ut = svd(D, full_matrices=False)
    U = Ut.T
    
    if n_components is None:
        n_components = min(nv, nt)
    n_components = min(n_components, len(s_all))
    
    return V[:, :n_components], s_all[:n_components], U[:, :n_components]


# =============================================================================
# SIGN CORRECTION
# =============================================================================

def align_signs(V_ref, U_ref, V_target, U_target, corr_threshold=-0.5):
    """
    Align signs of target SVD components to match reference.
    Detects cases needing twinx (scale inversion) vs sign flip.
    
    Parameters
    ----------
    V_ref : ndarray
        Reference spectra matrix
    U_ref : ndarray
        Reference time-traces matrix
    V_target : ndarray
        Target spectra matrix to align
    U_target : ndarray
        Target time-traces matrix to align
    corr_threshold : float, optional
        Correlation threshold for detecting misalignment (default: -0.5)
    
    Returns
    -------
    V_aligned : ndarray
        Sign-aligned spectra
    U_aligned : ndarray
        Sign-aligned time-traces
    info : dict
        Dictionary containing:
        - 'signs_V': signs applied to V (+1 or -1)
        - 'signs_U': signs applied to U (+1 or -1)
        - 'twinx_V': list of V indices needing twinx for plotting
        - 'twinx_U': list of U indices needing twinx for plotting
        - 'corr_V': correlation coefficients for V
        - 'corr_U': correlation coefficients for U
    
    Notes
    -----
    When correlation is negative but both vectors have the same sign
    (all positive or all negative), a sign flip would produce incorrect
    results. These cases are flagged for twinx plotting instead.
    """
    n_components = min(V_ref.shape[1], V_target.shape[1])
    
    V_aligned = V_target.copy()
    U_aligned = U_target.copy()
    signs_V = np.ones(n_components)
    signs_U = np.ones(n_components)
    twinx_V = []
    twinx_U = []
    corr_V = np.zeros(n_components)
    corr_U = np.zeros(n_components)
    
    for i in range(n_components):
        # Check correlation of spectra
        corr_V[i] = np.corrcoef(V_ref[:, i], V_target[:, i])[0, 1]
        # Check correlation of time-traces  
        corr_U[i] = np.corrcoef(U_ref[:, i], U_target[:, i])[0, 1]
        
        # Check if vectors have same sign (all positive or all negative)
        # Using small tolerance for numerical precision
        tol = 1e-10
        v_ref_same_sign = np.all(V_ref[:, i] >= -tol) or np.all(V_ref[:, i] <= tol)
        v_tgt_same_sign = np.all(V_target[:, i] >= -tol) or np.all(V_target[:, i] <= tol)
        u_ref_same_sign = np.all(U_ref[:, i] >= -tol) or np.all(U_ref[:, i] <= tol)
        u_tgt_same_sign = np.all(U_target[:, i] >= -tol) or np.all(U_target[:, i] <= tol)
        
        # V component - only act on strong correlations
        if corr_V[i] < corr_threshold and abs(corr_V[i]) >= 0.7:
            if v_ref_same_sign and v_tgt_same_sign:
                # Both same sign → need twinx for plotting
                twinx_V.append(i)
            else:
                # Can flip sign
                V_aligned[:, i] *= -1
                signs_V[i] = -1
        
        # U component - only act on strong correlations
        if corr_U[i] < corr_threshold and abs(corr_U[i]) >= 0.7:
            if u_ref_same_sign and u_tgt_same_sign:
                # Both same sign → need twinx for plotting
                twinx_U.append(i)
            else:
                # Can flip sign
                U_aligned[:, i] *= -1
                signs_U[i] = -1
    
    info = {
        'signs_V': signs_V,
        'signs_U': signs_U,
        'twinx_V': twinx_V,
        'twinx_U': twinx_U,
        'corr_V': corr_V,
        'corr_U': corr_U
    }
    
    return V_aligned, U_aligned, info


# =============================================================================
# COMPONENT NUMBER ESTIMATION
# =============================================================================

def estimate_n_components(s, method='ratio', threshold=10):
    """
    Estimate number of significant components from singular values.
    
    Parameters
    ----------
    s : ndarray
        Singular values (sorted descending)
    method : str, optional
        Method: 'ratio' (consecutive ratio), 'noise_fit' (exponential fit)
    threshold : float, optional
        Threshold for detection (default: 10 for ratio method)
    
    Returns
    -------
    n_components : int
        Estimated number of significant components
    """
    s = np.asarray(s)
    
    if method == 'ratio':
        # Find where consecutive ratio drops below threshold
        ratios = s[:-1] / (s[1:] + 1e-10)
        for i, r in enumerate(ratios):
            if r < threshold and i > 0:
                return i + 1
        return len(s)
    
    elif method == 'noise_fit':
        # Fit exponential to noise tail
        log_s = np.log(s + 1e-10)
        n = len(s)
        
        # Fit to last 60% of components
        fit_start = int(n * 0.4)
        if fit_start < 3:
            fit_start = 3
        
        idx = np.arange(fit_start, n)
        if len(idx) < 2:
            return n
        
        p = np.polyfit(idx, log_s[fit_start:], 1)
        fit_line = np.polyval(p, np.arange(n))
        
        # Signal components are above fit line + margin
        margin = 0.5
        n_components = np.sum(log_s > fit_line + margin)
        
        return max(1, n_components)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def plot_svd_comparison(wavenumbers, times, D, V_std, U_std, s_std, 
                        V_enh, U_enh, s_enh, n_components=5,
                        twinx_V=None, twinx_U=None,
                        figsize=(18, 12), savepath=None):
    """
    Create Figure 3-style comparison plot.
    
    Parameters
    ----------
    wavenumbers : ndarray
        Wavenumber axis
    times : ndarray
        Time axis
    D : ndarray
        Original data matrix
    V_std, U_std, s_std : ndarray
        Standard SVD results
    V_enh, U_enh, s_enh : ndarray
        Enhanced SVD results
    n_components : int, optional
        Number of components to plot (default: 5)
    twinx_V : list, optional
        Indices of V components needing twinx
    twinx_U : list, optional
        Indices of U components needing twinx
    figsize : tuple, optional
        Figure size
    savepath : str, optional
        Path to save figure
    
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    if twinx_V is None:
        twinx_V = []
    if twinx_U is None:
        twinx_U = []
    
    fig = plt.figure(figsize=figsize)
    
    # ROW A: Overview
    ax_3d = fig.add_subplot(3, 5, (1, 2), projection='3d')
    T, W = np.meshgrid(times, wavenumbers)
    ax_3d.plot_surface(T, W, D, cmap='RdBu_r', alpha=0.9, linewidth=0)
    ax_3d.set_xlabel('Time')
    ax_3d.set_ylabel('Wavenumber (cm⁻¹)')
    ax_3d.set_zlabel('ΔAbs')
    ax_3d.set_title('(A1) 3D Data', fontweight='bold')
    ax_3d.view_init(elev=25, azim=-60)
    
    ax_2d = fig.add_subplot(3, 5, (3, 4))
    im = ax_2d.imshow(D, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], wavenumbers[-1], wavenumbers[0]],
                       cmap='RdBu_r')
    ax_2d.set_xlabel('Time')
    ax_2d.set_ylabel('Wavenumber (cm⁻¹)')
    ax_2d.set_title('(A2) 2D Data', fontweight='bold')
    plt.colorbar(im, ax=ax_2d, label='ΔAbs', shrink=0.8)
    
    ax_sv = fig.add_subplot(3, 5, 5)
    n_plot = min(20, len(s_std), len(s_enh))
    idx = np.arange(1, n_plot + 1)
    ax_sv.semilogy(idx, s_std[:n_plot], 'bo-', markersize=6, label='Standard')
    ax_sv.semilogy(idx, s_enh[:n_plot], 'rs-', markersize=6, label='Enhanced')
    ax_sv.set_xlabel('Index')
    ax_sv.set_ylabel('Singular Value')
    ax_sv.set_title('(A3) Singular Values', fontweight='bold')
    ax_sv.legend(fontsize=8)
    ax_sv.grid(True, alpha=0.3)
    
    # ROW B: Spectra
    for i in range(n_components):
        ax = fig.add_subplot(3, 5, 6 + i)
        v_std_norm = V_std[:, i] / (np.max(np.abs(V_std[:, i])) + 1e-10)
        v_enh_norm = V_enh[:, i] / (np.max(np.abs(V_enh[:, i])) + 1e-10)
        
        if i in twinx_V:
            # Use twinx for scale inversion
            line1, = ax.plot(wavenumbers, v_std_norm, 'k-', linewidth=1.2, label='Raw SVD')
            ax2 = ax.twinx()
            line2, = ax2.plot(wavenumbers, v_enh_norm, 'g-', linewidth=1.2, alpha=0.8, label='Enhanced')
            ax2.invert_yaxis()
            ax2.set_yticks([])
            ax.legend([line1, line2], ['Raw SVD', 'Enhanced'], fontsize=7, loc='best')
        else:
            ax.plot(wavenumbers, v_std_norm, 'k-', linewidth=1.2, label='Raw SVD')
            ax.plot(wavenumbers, v_enh_norm, 'g-', linewidth=1.2, alpha=0.8, label='Enhanced')
            if i == 0:
                ax.legend(fontsize=7)
        
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=8)
        ax.set_title(f'(B{i+1}) V{i+1}', fontweight='bold')
        ax.invert_xaxis()
        if i == 0:
            ax.set_ylabel('Normalized', fontsize=9)
    
    # ROW C: Time-traces
    for i in range(n_components):
        ax = fig.add_subplot(3, 5, 11 + i)
        u_std_norm = U_std[:, i] / (np.max(np.abs(U_std[:, i])) + 1e-10)
        u_enh_norm = U_enh[:, i] / (np.max(np.abs(U_enh[:, i])) + 1e-10)
        
        if i in twinx_U:
            # Use twinx for scale inversion
            line1, = ax.plot(times, u_std_norm, 'k-', linewidth=1.2, label='Raw SVD')
            ax2 = ax.twinx()
            line2, = ax2.plot(times, u_enh_norm, 'g-', linewidth=1.2, alpha=0.8, label='Enhanced')
            ax2.invert_yaxis()
            ax2.set_yticks([])
            ax.legend([line1, line2], ['Raw SVD', 'Enhanced'], fontsize=7, loc='best')
        else:
            ax.plot(times, u_std_norm, 'k-', linewidth=1.2, label='Raw SVD')
            ax.plot(times, u_enh_norm, 'g-', linewidth=1.2, alpha=0.8, label='Enhanced')
            if i == 0:
                ax.legend(fontsize=7)
        
        ax.set_xlabel('Time', fontsize=8)
        ax.set_title(f'(C{i+1}) U{i+1}', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Normalized', fontsize=9)
    
    plt.suptitle('SVD Baseline Fluctuation Removal - Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def process_spectroscopy_data(D, wavenumbers=None, times=None, n_components=5,
                               apod_start=0.10, apod_end=0.30, 
                               auto_sign=True, plot=False, savepath=None):
    """
    Complete processing pipeline for time-resolved spectroscopy data.
    
    Parameters
    ----------
    D : ndarray
        Data matrix (nv × nt)
    wavenumbers : ndarray, optional
        Wavenumber axis (for plotting)
    times : ndarray, optional
        Time axis (for plotting)
    n_components : int, optional
        Number of components to extract (default: 5)
    apod_start : float, optional
        FFT apodization start (default: 0.10)
    apod_end : float, optional
        FFT apodization end (default: 0.30)
    auto_sign : bool, optional
        Automatically align signs (default: True)
    plot : bool, optional
        Generate comparison plot (default: False)
    savepath : str, optional
        Path to save plot
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'V_std', 'S_std', 'U_std': Standard SVD results
        - 'V_enh', 'S_enh', 'U_enh': Enhanced SVD results
        - 'n_estimated': Estimated number of components
        - 'align_info': Sign alignment info (if auto_sign=True)
        - 'fig': Figure object (if plot=True)
    """
    D = np.asarray(D, dtype=float)
    nv, nt = D.shape
    
    # Standard SVD
    V_std, S_std, U_std = standard_svd(D, n_components=n_components)
    
    # Enhanced SVD
    V_enh, S_enh, U_enh, info = enhanced_svd(
        D, n_components=n_components,
        apod_start=apod_start, apod_end=apod_end,
        return_all=True
    )
    
    # Auto sign alignment
    align_info = None
    twinx_V = []
    twinx_U = []
    if auto_sign:
        V_enh, U_enh, align_info = align_signs(V_std, U_std, V_enh, U_enh)
        twinx_V = align_info['twinx_V']
        twinx_U = align_info['twinx_U']
    
    # Estimate number of components
    n_estimated = estimate_n_components(info['s_all'])
    
    # Get all singular values for standard SVD
    _, s_all_std, _ = svd(D, full_matrices=False)
    
    results = {
        'V_std': V_std,
        'S_std': S_std,
        'U_std': U_std,
        'V_enh': V_enh,
        'S_enh': S_enh,
        'U_enh': U_enh,
        's_all_std': s_all_std,
        's_all_enh': info['s_all'],
        'n_estimated': n_estimated,
        'wv': info['wv'],
        'wt': info['wt'],
        'align_info': align_info
    }
    
    # Plot
    if plot:
        if wavenumbers is None:
            wavenumbers = np.arange(nv)
        if times is None:
            times = np.arange(nt)
        
        fig = plot_svd_comparison(
            wavenumbers, times, D,
            V_std, U_std, s_all_std,
            V_enh, U_enh, info['s_all'],
            n_components=n_components,
            twinx_V=twinx_V,
            twinx_U=twinx_U,
            savepath=savepath
        )
        results['fig'] = fig
    
    return results


# =============================================================================
# MAIN (EXAMPLE USAGE)
# =============================================================================

if __name__ == '__main__':
    import pandas as pd
    
    # Example usage
    print("SVD Baseline Fluctuation Removal Module")
    print("=" * 50)
    
    # Try to load example data
    try:
        df = pd.read_csv('/mnt/user-data/uploads/data_test_625_49.csv', index_col=0)
        wavenumbers = df.index.values.astype(float)
        times = df.columns.values.astype(float)
        D = df.values.astype(float)
        
        print(f"Loaded data: {D.shape[0]} wavenumbers × {D.shape[1]} time points")
        
        # Process
        results = process_spectroscopy_data(
            D, wavenumbers, times,
            n_components=5,
            auto_sign=True,
            plot=True,
            savepath='svd_baseline_removal_example.png'
        )
        
        print(f"\nEstimated components: {results['n_estimated']}")
        print(f"Standard SVD singular values: {results['S_std']}")
        print(f"Enhanced SVD singular values: {results['S_enh']}")
        print("\nFigure saved: svd_baseline_removal_example.png")
        
    except FileNotFoundError:
        print("No example data found. Module loaded successfully.")
        print("\nUsage:")
        print("  from svd_baseline_removal import enhanced_svd, process_spectroscopy_data")
        print("  V, S, U = enhanced_svd(D)")
        print("  results = process_spectroscopy_data(D, wavenumbers, times, plot=True)")
