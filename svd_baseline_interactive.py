#!/usr/bin/env python3
"""
Enhanced SVD for Signal-Noise Separation in Spectroscopic Data - Interactive Version
=======================================================

User-friendly interactive script for non-expert users.
Guides through file loading, visualization, and component alignment.

Based on: Lórenz-Fonfría & Kandori JACS 2009, 131, 5891-5901
          Kalka, A.J. & Turek, A.M. Appl. Spectrosc. 2023, 77, 426-432
Author: Nicolas Spegazzini
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svd_baseline_removal import (
    enhanced_svd, standard_svd, estimate_noise_weights,
    derivative_matrix, integrate_fft
)
import os


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print welcome header."""
    print("=" * 60)
    print("  SVD Baseline Fluctuation Removal - Interactive Mode")
    print("=" * 60)
    print()


def get_file_path():
    """Ask user for data file path."""
    print("STEP 1: Load Data")
    print("-" * 40)
    print("Enter the path to your CSV data file.")
    print("Format: wavenumbers in first column, times in first row")
    print()
    
    while True:
        filepath = input("File path: ").strip()
        
        # Remove quotes if user dragged file
        filepath = filepath.strip('"').strip("'")
        
        if os.path.exists(filepath):
            return filepath
        else:
            print(f"  ✗ File not found: {filepath}")
            print("  Please try again.\n")


def load_data(filepath):
    """Load CSV data file."""
    try:
        df = pd.read_csv(filepath, index_col=0)
        wavenumbers = df.index.values.astype(float)
        times = df.columns.values.astype(float)
        D = df.values.astype(float)
        
        print(f"  ✓ Loaded: {D.shape[0]} wavenumbers × {D.shape[1]} time points")
        print(f"  ✓ Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
        print(f"  ✓ Time range: {times.min():.1f} - {times.max():.1f}")
        print()
        
        return D, wavenumbers, times
    
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return None, None, None


def compute_svd(D, n_components=5):
    """Compute standard and enhanced SVD."""
    print("Computing SVD...")
    V_std, S_std, U_std = standard_svd(D, n_components=n_components)
    V_enh, S_enh, U_enh = enhanced_svd(D, n_components=n_components)
    print("  ✓ SVD complete")
    print()
    return V_std, S_std, U_std, V_enh, S_enh, U_enh


def plot_comparison(wavenumbers, times, D, V_std, U_std, S_std, V_enh, U_enh, S_enh,
                    flip_V, flip_U, twinx_V, twinx_U, n_components=5):
    """Generate comparison plot with user choices applied."""
    
    # Apply flips
    V_plot = V_enh.copy()
    U_plot = U_enh.copy()
    
    for i in flip_V:
        V_plot[:, i] *= -1
    for i in flip_U:
        U_plot[:, i] *= -1
    
    fig = plt.figure(figsize=(18, 12))
    
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
    
    # Get all singular values
    from scipy.linalg import svd as scipy_svd
    _, s_std_all, _ = scipy_svd(D, full_matrices=False)
    _, _, _, info = enhanced_svd(D, n_components=n_components, return_all=True)
    s_enh_all = info['s_all']
    
    ax_sv = fig.add_subplot(3, 5, 5)
    n_plot = min(20, len(s_std_all), len(s_enh_all))
    idx = np.arange(1, n_plot + 1)
    ax_sv.semilogy(idx, s_std_all[:n_plot], 'bo-', markersize=6, label='Standard')
    ax_sv.semilogy(idx, s_enh_all[:n_plot], 'rs-', markersize=6, label='Enhanced')
    ax_sv.set_xlabel('Index')
    ax_sv.set_ylabel('Singular Value')
    ax_sv.set_title('(A3) Singular Values', fontweight='bold')
    ax_sv.legend(fontsize=8)
    ax_sv.grid(True, alpha=0.3)
    
    # ROW B: Spectra
    for i in range(n_components):
        ax = fig.add_subplot(3, 5, 6 + i)
        v_std_norm = V_std[:, i] / (np.max(np.abs(V_std[:, i])) + 1e-10)
        v_enh_norm = V_plot[:, i] / (np.max(np.abs(V_plot[:, i])) + 1e-10)
        
        if i in twinx_V:
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
        u_enh_norm = U_plot[:, i] / (np.max(np.abs(U_plot[:, i])) + 1e-10)
        
        if i in twinx_U:
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
    
    return fig


def interactive_alignment(V_std, U_std, V_enh, U_enh, n_components=5):
    """Interactive component alignment."""
    
    print("\nSTEP 3: Component Alignment")
    print("-" * 40)
    print("For each component, compare black (Raw) vs green (Enhanced).")
    print("Choose how to align the GREEN line:\n")
    print("  [F] Flip   - if green crosses zero (has + and - values)")
    print("  [T] Twinx  - if green is all positive or all negative")  
    print("  [S] Skip   - leave unchanged (default)\n")
    
    flip_V = []
    flip_U = []
    twinx_V = []
    twinx_U = []
    
    # Process V components (spectra)
    print("── SPECTRA (V) ──")
    for i in range(n_components):
        r = np.corrcoef(V_std[:, i], V_enh[:, i])[0, 1]
        v_min = V_enh[:, i].min()
        v_max = V_enh[:, i].max()
        crosses_zero = (v_min < 0) and (v_max > 0)
        
        print(f"\n  V{i+1}: correlation r = {r:+.3f}")
        print(f"       green range: [{v_min:.3f}, {v_max:.3f}]")
        if crosses_zero:
            print("       → crosses zero (mixed signs)")
        else:
            print("       → all same sign")
        
        if r < -0.5:
            print("       ⚠ Negative correlation - likely needs alignment")
        
        choice = input(f"  V{i+1} action [F/T/S, default=S]: ").strip().upper()
        
        if choice == 'F':
            flip_V.append(i)
            print("       → Will FLIP")
        elif choice == 'T':
            twinx_V.append(i)
            print("       → Will TWINX")
        else:
            print("       → Skipped")
    
    # Process U components (time-traces)
    print("\n── TIME-TRACES (U) ──")
    for i in range(n_components):
        r = np.corrcoef(U_std[:, i], U_enh[:, i])[0, 1]
        u_min = U_enh[:, i].min()
        u_max = U_enh[:, i].max()
        crosses_zero = (u_min < 0) and (u_max > 0)
        
        print(f"\n  U{i+1}: correlation r = {r:+.3f}")
        print(f"       green range: [{u_min:.3f}, {u_max:.3f}]")
        if crosses_zero:
            print("       → crosses zero (mixed signs)")
        else:
            print("       → all same sign")
        
        if r < -0.5:
            print("       ⚠ Negative correlation - likely needs alignment")
        
        choice = input(f"  U{i+1} action [F/T/S, default=S]: ").strip().upper()
        
        if choice == 'F':
            flip_U.append(i)
            print("       → Will FLIP")
        elif choice == 'T':
            twinx_U.append(i)
            print("       → Will TWINX")
        else:
            print("       → Skipped")
    
    return flip_V, flip_U, twinx_V, twinx_U


def main():
    """Main interactive workflow."""
    
    clear_screen()
    print_header()
    
    # Step 1: Get file path
    filepath = get_file_path()
    D, wavenumbers, times = load_data(filepath)
    
    if D is None:
        print("Failed to load data. Exiting.")
        return
    
    # Compute SVD
    n_components = 5
    V_std, S_std, U_std, V_enh, S_enh, U_enh = compute_svd(D, n_components)
    
    # Initialize alignment choices
    flip_V = []
    flip_U = []
    twinx_V = []
    twinx_U = []
    
    # Main loop
    while True:
        # Step 2: Generate plot
        print("\nSTEP 2: Generating plot...")
        print("Close the plot window when done inspecting.\n")
        
        fig = plot_comparison(
            wavenumbers, times, D,
            V_std, U_std, S_std,
            V_enh, U_enh, S_enh,
            flip_V, flip_U, twinx_V, twinx_U,
            n_components
        )
        plt.show()
        
        # Step 4: Ask if satisfied
        print("\n" + "-" * 40)
        satisfied = input("Are you satisfied with the alignment? [Y/N]: ").strip().upper()
        
        if satisfied == 'Y':
            # Save figure
            output_dir = os.path.dirname(filepath) or '.'
            output_name = os.path.splitext(os.path.basename(filepath))[0] + '_SVD_result.png'
            output_path = os.path.join(output_dir, output_name)
            
            # Regenerate and save
            fig = plot_comparison(
                wavenumbers, times, D,
                V_std, U_std, S_std,
                V_enh, U_enh, S_enh,
                flip_V, flip_U, twinx_V, twinx_U,
                n_components
            )
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\n  ✓ Figure saved: {output_path}")
            print("\nDone! Thank you for using SVD Baseline Removal.")
            break
        
        else:
            # Step 3: Interactive alignment
            flip_V, flip_U, twinx_V, twinx_U = interactive_alignment(
                V_std, U_std, V_enh, U_enh, n_components
            )
            print("\nApplying your choices...")


if __name__ == '__main__':
    main()
