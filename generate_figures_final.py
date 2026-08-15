"""
generate_figures_final.py
=========================
Generates Figures 1–4 for:
  "Enhanced SVD for Signal–Noise Separation in Spectroscopic Data"
  N. Spegazzini — Applied Spectroscopy (Note, ASP-26-0104.R1)

Requirements (same folder):
  - svd_baseline_removal.py
  - data_test_625_49.csv  (Figures 2-4 only; Figure 1 is generated from the
    seeded synthetic model and needs no external data file)

Output: Figure1_synthetic.tiff, Figure2_experimental.tiff,
        Figure3_abstract_spectra.tiff, Figure4_abstract_timetraces.tiff
Format: TIFF LZW, 800 dpi, 7.0 in wide
Colors: Standard SVD = black, Enhanced SVD = orange (#E07B00)
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.linalg import svd as scipy_svd
from svd_baseline_removal import standard_svd, enhanced_svd, align_signs, derivative_matrix

# ── Output directory (same as script by default) ──────────────────────────────
OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUTDIR, 'data_test_625_49.csv')

# ── Style ──────────────────────────────────────────────────────────────────────
DPI   = 800
COL2  = 7.0
C_STD = 'black'
C_ENH = '#E07B00'   # dark orange

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'lines.linewidth': 1.0, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    # Flatten to RGB (no alpha) for ScholarOne compatibility
    from PIL import Image
    im = Image.open(path)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        im_rgba = im.convert('RGBA')
        bg.paste(im_rgba, mask=im_rgba.split()[3])
        bg.save(path, compression='tiff_lzw', dpi=(DPI, DPI))
    else:
        im.load()
        im.save(path, compression='tiff_lzw', dpi=(DPI, DPI))
    chk = Image.open(path)
    print(f"  Saved: {name}  [{chk.mode}, {chk.size[0]}x{chk.size[1]}, "
          f"{chk.info.get('dpi')} dpi]")


# ── Synthetic data generator ───────────────────────────────────────────────────
def make_synthetic(seed=42, nv=200, nt=40, snr=50, sigma=20, amp=0.25):
    rng = np.random.default_rng(seed)
    wn = np.linspace(900, 1800, nv)
    t  = np.linspace(0, 1, nt)
    s1 = np.exp(-((wn - 1200) / sigma)**2)
    s2 = np.exp(-((wn - 1650) / sigma)**2)
    c1 = np.exp(-t / 0.3); c2 = 1 - np.exp(-t / 0.3)
    D  = np.outer(s1, c1) + np.outer(s2, c2)
    x  = (wn - wn[0]) / (wn[-1] - wn[0])
    D += amp * np.sin(2*np.pi*(x[:, None] + 0.5*t[None, :]))
    D += rng.normal(0, np.max(D) / snr, D.shape)
    return D, wn, t


# ══════════════════════════════════════════════════════════════════════════════
print("Computing decompositions...")

# Synthetic
D_syn, wn_syn, t_syn = make_synthetic()
_, s_std_s, _ = scipy_svd(D_syn, full_matrices=False)
_, _, _, info_s = enhanced_svd(D_syn, n_components=10, return_all=True)
s_enh_s   = info_s['s_all']
D_prime_s = derivative_matrix(D_syn)

# Real experimental — skipped if the data file is absent
HAVE_DATA = os.path.exists(DATA_PATH)
if HAVE_DATA:
    df   = pd.read_csv(DATA_PATH, index_col=0)
    wn   = df.index.values.astype(float)
    t    = df.columns.values.astype(float)
    D    = df.values.astype(float)
    V_std, S_std, U_std = standard_svd(D, n_components=10)
    V_enh, S_enh, U_enh, info = enhanced_svd(D, n_components=10, return_all=True)
    V_enh, U_enh, _ = align_signs(V_std, U_std, V_enh, U_enh)
    D_prime    = derivative_matrix(D)
    _, s_all_std, _ = scipy_svd(D, full_matrices=False)
    s_all_enh  = info['s_all']
else:
    print(f"\n  {os.path.basename(DATA_PATH)} not found.")
    print("  Figure 1 (synthetic) will be generated; Figures 2-4 require the")
    print("  experimental data, available from the author on reasonable request.\n")

n_plot = 12
idx = np.arange(1, n_plot + 1)

# ══ FIGURE 1 — Synthetic (3-panel) ════════════════════════════════════════════
print("Figure 1...")
fig1, axes1 = plt.subplots(1, 3, figsize=(COL2, 2.4))

ax = axes1[0]
im = ax.imshow(D_syn, aspect='auto', origin='upper',
               extent=[t_syn[0], t_syn[-1], wn_syn[-1], wn_syn[0]],
               cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel('Time (arb. units)'); ax.set_ylabel('Wavenumber (cm⁻¹)')
ax.set_title('(a) Data matrix D')
cb = fig1.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
cb.ax.tick_params(labelsize=7); cb.set_label('Absorbance', fontsize=7)

ax = axes1[1]
for j in range(D_prime_s.shape[1]):
    ax.plot(wn_syn, D_prime_s[:, j], color='black', lw=0.5, alpha=0.35)
ax.set_xlabel('Wavenumber (cm⁻¹)'); ax.set_ylabel("d(Abs)/dν (arb. units)")
ax.set_title("(b) Derivative spectra D'")
ax.axhline(0, color='gray', lw=0.4, ls='--')
ax.yaxis.set_major_locator(MaxNLocator(4))

ax = axes1[2]
ax.semilogy(idx, s_std_s[:n_plot], color=C_STD, marker='o', ms=4, lw=1.0, label='Standard SVD')
ax.semilogy(idx, s_enh_s[:n_plot], color=C_ENH, marker='s', ms=4, lw=1.0, ls='--', label='Enhanced SVD')
ax.axvline(2.5, color=C_ENH, lw=0.8, ls=':')
ax.annotate('k = 2', xy=(2.5, s_std_s[3]), xytext=(5.0, s_std_s[3] * 2.2),
            fontsize=7, color=C_ENH,
            arrowprops=dict(arrowstyle='->', color=C_ENH, lw=0.8))
ax.set_xlabel('Component index'); ax.set_ylabel('Singular value')
ax.set_title('(c) Singular values'); ax.legend(loc='upper right')
ax.set_xticks(range(1, n_plot + 1, 2)); ax.grid(True, which='both', alpha=0.2, lw=0.4)

fig1.tight_layout(pad=0.5, w_pad=2.2)
save(fig1, 'Figure1_synthetic.tiff')

if not HAVE_DATA:
    print("\nFigure 1 complete. Figures 2-4 skipped (no experimental data).")
    sys.exit(0)

# ══ FIGURE 2 — Experimental FTIR (3-panel) ════════════════════════════════════
print("Figure 2...")
fig2, axes2 = plt.subplots(1, 3, figsize=(COL2, 2.4))

ax = axes2[0]
im2 = ax.imshow(D, aspect='auto', origin='upper',
                extent=[t[0], t[-1], wn[-1], wn[0]],
                cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel('Time (min)'); ax.set_ylabel('Wavenumber (cm⁻¹)')
ax.set_title('(a) Data matrix D')
cb2 = fig2.colorbar(im2, ax=ax, shrink=0.85, pad=0.04)
cb2.ax.tick_params(labelsize=7); cb2.set_label('Absorbance', fontsize=7)

ax = axes2[1]
grays = plt.cm.Greys(np.linspace(0.25, 0.85, D_prime.shape[1]))
for j in range(D_prime.shape[1]):
    ax.plot(wn, D_prime[:, j], color=grays[j], lw=0.5)
ax.set_xlabel('Wavenumber (cm⁻¹)'); ax.set_ylabel("d(Abs)/dν (arb. units)")
ax.set_title("(b) Derivative spectra D'")
ax.axhline(0, color='gray', lw=0.4, ls='--')
ax.yaxis.set_major_locator(MaxNLocator(4)); ax.invert_xaxis()

ax = axes2[2]
ax.semilogy(idx, s_all_std[:n_plot], color=C_STD, marker='o', ms=4, lw=1.0, label='Standard SVD')
ax.semilogy(idx, s_all_enh[:n_plot], color=C_ENH, marker='s', ms=4, lw=1.0, ls='--', label='Enhanced SVD')
ax.set_xlabel('Component index'); ax.set_ylabel('Singular value')
ax.set_title('(c) Singular values'); ax.legend(loc='upper right')
ax.set_xticks(range(1, n_plot + 1, 2)); ax.grid(True, which='both', alpha=0.2, lw=0.4)

fig2.tight_layout(pad=0.5, w_pad=2.2)
save(fig2, 'Figure2_experimental.tiff')

# ══ FIGURE 3 — Abstract spectra V1–V4 (2×2) ══════════════════════════════════
print("Figure 3...")
V_enh_f = V_enh.copy()
V_enh_f[:, 3] *= -1   # flip V4

fig3, axes3 = plt.subplots(2, 2, figsize=(COL2, 4.2))
for i, ax in enumerate(axes3.flatten()):
    v_s = V_std[:, i] / (np.max(np.abs(V_std[:, i])) + 1e-12)
    v_e = V_enh_f[:, i] / (np.max(np.abs(V_enh_f[:, i])) + 1e-12)
    ax.plot(wn, v_s, color=C_STD, lw=1.0, label='Standard SVD')
    ax.plot(wn, v_e, color=C_ENH, lw=1.0, ls='--', label='Enhanced SVD')
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    ax.set_title(f'V{i+1}'); ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Normalized intensity')
    ax.yaxis.set_major_locator(MaxNLocator(4)); ax.invert_xaxis()
    if i == 0: ax.legend(loc='best')

fig3.tight_layout(pad=0.6)
save(fig3, 'Figure3_abstract_spectra.tiff')

# ══ FIGURE 4 — Abstract time-traces U1–U4 (2×2) ══════════════════════════════
print("Figure 4...")
U_enh_f = U_enh.copy()
U_enh_f[:, 0] *= -1   # flip U1
U_enh_f[:, 3] *= -1   # flip U4

fig4, axes4 = plt.subplots(2, 2, figsize=(COL2, 4.2))
for i, ax in enumerate(axes4.flatten()):
    u_s = U_std[:, i] / (np.max(np.abs(U_std[:, i])) + 1e-12)
    u_e = U_enh_f[:, i] / (np.max(np.abs(U_enh_f[:, i])) + 1e-12)

    if i == 0:
        # plotyy style: independent y-axes, right spine hidden
        u_s_sc = (u_s - u_s.min()) / (u_s.max() - u_s.min())
        u_e_sc = (u_e - u_e.min()) / (u_e.max() - u_e.min())
        ax.plot(t, u_s_sc, color=C_STD, lw=1.0)
        ax2 = ax.twinx()
        ax2.plot(t, u_e_sc, color=C_ENH, lw=1.0, ls='--')
        ax2.set_yticks([]); ax2.spines['right'].set_visible(False)
        lines = [plt.Line2D([0],[0], color=C_STD, lw=1.0),
                 plt.Line2D([0],[0], color=C_ENH, lw=1.0, ls='--')]
        ax.legend(lines, ['Standard SVD', 'Enhanced SVD'], loc='upper left', fontsize=7)
        ax.set_ylabel('Normalized amplitude')
    else:
        ax.plot(t, u_s, color=C_STD, lw=1.0, label='Standard SVD')
        ax.plot(t, u_e, color=C_ENH, lw=1.0, ls='--', label='Enhanced SVD')
        ax.axhline(0, color='gray', lw=0.4, ls=':')
        ax.set_ylabel('Normalized amplitude')
        if i == 1: ax.legend(loc='best', fontsize=7)

    ax.set_title(f'U{i+1}'); ax.set_xlabel('Time (min)')
    ax.yaxis.set_major_locator(MaxNLocator(4))

fig4.tight_layout(pad=0.6)
save(fig4, 'Figure4_abstract_timetraces.tiff')

print("\nAll figures complete.")
