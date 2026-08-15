"""
rank_analysis.py
================
Rank determination and subspace projection analysis for the experimental
FTIR dataset in:
  "Enhanced SVD for Signal-Noise Separation in Spectroscopic Data"
  N. Spegazzini - Applied Spectroscopy (Note)

Produces the singular values, successive ratios and subspace projections
reported in the Experimental FTIR Spectra section.

Requires the experimental data file, which is available from the author on
reasonable request. The precomputed output is provided as rank_results.txt.
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
from svd_baseline_removal import enhanced_svd

df = pd.read_csv('data_test_625_49.csv', index_col=0)
wn = df.index.values.astype(float)
D = df.values.astype(float)

A0 = D[:, 0]
Ac = D - D.mean(axis=1, keepdims=True)
dA = D - A0[:, None]


def ratios(s, n=5):
    return [s[i] / s[i + 1] for i in range(n)]


print(f'Data matrix: {D.shape[0]} x {D.shape[1]}\n')

print('Singular values')
s_std = svd(D, full_matrices=False)[1]
s_enh = enhanced_svd(D, n_components=6, return_all=True)[3]['s_all']
s_ac = svd(Ac, full_matrices=False)[1]
s_da = svd(dA, full_matrices=False)[1]
print(f'{"i":>2} {"raw A std":>12} {"raw A enh":>12} {"centered":>12} {"dA":>12}')
for i in range(6):
    print(f'{i+1:>2} {s_std[i]:12.4f} {s_enh[i]:12.5f} {s_ac[i]:12.4f} {s_da[i]:12.4f}')

print('\nSuccessive ratios s_i / s_(i+1)')
print(f'{"pos":>5} {"raw A std":>12} {"raw A enh":>12} {"centered":>12} {"dA":>12}')
for i, (a, b, c, d) in enumerate(zip(ratios(s_std), ratios(s_enh),
                                     ratios(s_ac, 5), ratios(s_da, 5))):
    print(f'{i+1}->{i+2:<2} {a:12.2f} {b:12.2f} {c:12.2f} {d:12.2f}')

print('\nSubspace projections (standard SVD on raw A)')
print('Cosines of angles between unit vectors, not singular values.')
V = svd(D, full_matrices=False)[0]
P = np.linalg.qr(svd(dA, full_matrices=False)[0][:, :2])[0]
a0 = A0 / np.linalg.norm(A0)
print(f'{"comp":>5} {"proj on dA (2-D)":>18} {"proj on A0":>12}')
for i in range(4):
    v = V[:, i]
    print(f'V{i+1:<4} {np.linalg.norm(P.T @ v):18.3f} {abs(v @ a0):12.3f}')

Ve = enhanced_svd(D, n_components=6, return_all=True)[0]
v1 = Ve[:, 0] / np.linalg.norm(Ve[:, 0])
print(f'\nEnhanced V1 projection onto A0: {abs(v1 @ a0):.3f}')
print('The derivative removes constant offsets but not a structured')
print('invariant background; centering or difference construction does.')

m = D.mean(axis=1)
print(f'\nMean spectrum maximum: A = {m.max():.2f} at {wn[np.argmax(m)]:.1f} cm-1')
