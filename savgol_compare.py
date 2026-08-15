import numpy as np, pandas as pd
from scipy.linalg import svd
from scipy.signal import savgol_filter
from svd_baseline_removal import enhanced_svd

df = pd.read_csv('data_test_625_49.csv', index_col=0)
wn = df.index.values.astype(float)
D = df.values.astype(float)

def sv(M, n=6):
    _, s, _ = svd(M, full_matrices=False)
    return s[:n]

def ratios(s):
    return [s[k] / s[k + 1] for k in range(len(s) - 1)]

def show(tag, s):
    print(f'{tag:34s}', ' '.join(f'{v:9.4g}' for v in s))
    print(f'{"  ratios":34s}', ' '.join(f'{r:9.2f}' for r in ratios(s)))

# Enhanced SVD reference
_, _, _, info = enhanced_svd(D, n_components=6, return_all=True)
show('enhanced SVD (this work)', info['s_all'][:6])
print()

# SavGol derivatives, several window/order settings, then PCA (mean-centered)
for order, dname in [(1, '1st deriv'), (2, '2nd deriv')]:
    for w, p in [(7, 2), (11, 3), (15, 3), (21, 4), (31, 4)]:
        Dd = savgol_filter(D, window_length=w, polyorder=p, deriv=order, axis=0)
        Dc = Dd - Dd.mean(axis=1, keepdims=True)     # PCA = mean-centered
        show(f'SavGol {dname} w={w:2d} p={p} + PCA', sv(Dc))
    print()

# How much do the retained time-trace subspaces agree?
# Compared in the temporal domain (U), which is common to both methods.
# The spectral domain is NOT comparable: enhanced V is integrated back to the
# absorbance domain, whereas SavGol V remains in the derivative domain.
_, _, Ue, _ = enhanced_svd(D, n_components=2, return_all=True)
Qe = np.linalg.qr(Ue[:, :2])[0]


def principal_angles(A, B):
    sing = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.sort(np.degrees(np.arccos(np.clip(sing, -1, 1))))


print('Principal angles, 2-D time-trace subspace vs enhanced SVD')
print('  (a) uncentered SVD on the SavGol derivative -- like-for-like')
for order, dname in [(1, '1st'), (2, '2nd')]:
    for w, p in [(7, 2), (11, 3), (15, 3), (21, 4), (31, 4)]:
        Dd = savgol_filter(D, window_length=w, polyorder=p, deriv=order, axis=0)
        Us = svd(Dd, full_matrices=False)[2].T
        a = principal_angles(Qe, np.linalg.qr(Us[:, :2])[0])
        print(f'    SavGol {dname} w={w:2d} p={p}: {a[0]:5.1f} deg, {a[1]:5.1f} deg')

print('  (b) mean-centered across time (PCA convention)')
for order, dname in [(1, '1st'), (2, '2nd')]:
    for w, p in [(7, 2), (11, 3), (15, 3), (21, 4), (31, 4)]:
        Dd = savgol_filter(D, window_length=w, polyorder=p, deriv=order, axis=0)
        Dc = Dd - Dd.mean(axis=1, keepdims=True)
        Us = svd(Dc, full_matrices=False)[2].T
        a = principal_angles(Qe, np.linalg.qr(Us[:, :2])[0])
        print(f'    SavGol {dname} w={w:2d} p={p}: {a[0]:5.1f} deg, {a[1]:5.1f} deg')
