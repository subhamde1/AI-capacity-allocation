Python 3.12; NumPy 2.3.5; SciPy 1.17.0 for the MILP check.
Prints the 27-cell design, policy summaries, audits, and sensitivity.
"""
from itertools import product
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

D = np.array([2, 4, 2, 4, 4, 2, 4, 2])
V = np.array([12, 15, 13, 16, 17, 14, 18, 11])
H = np.array([0, 1, 1, 0, 0, 1, 1, 0])
M = np.array(list(product(range(4), repeat=8)), dtype=np.int8)
AD = M > 0
Q = np.choose(M, [np.zeros(8), np.zeros(8), D / 2, D])
VALUE = AD @ V
MASK = AD @ (1 << np.arange(8))
FIRM = np.all(M < 2, axis=1)
HALF = np.all(M < 3, axis=1)
s = 20260922
costs = []
for r in range(128):
    row = []
    for i in range(8):
        s = (1664525 * s + 1013904223) % (2 ** 32)
        row.append((5 * s) // (2 ** 32))
    costs.append(row)
costs = np.array(costs)


def loads(kappa):
    a = np.zeros((8, 4, 4))
    for i in range(8):
        alpha = kappa + (1 - kappa) * H[i]
        for m, q in [(1, 0), (2, D[i] / 2), (3, D[i])]:
            a[i, m, i % 2] = D[i] - q
            a[i, m, 2] = alpha * q
            a[i, m, 3] = (1 - alpha) * q
    total = sum(a[i, M[:, i]] for i in range(8))
    return a, total


def best(ids, w):
    return int(ids[np.argmax(w[ids])])


def pay_util(j, w, ids, theta):
    p = np.zeros(8)
    for i in range(8):
        absent = ids[M[ids, i] == 0]
        wm = w[best(absent, w)]
        own = V[i] * AD[j, i] - theta[i] * Q[j, i]
        p[i] = wm - (w[j] - own)
    u = V * AD[j] - theta * Q[j] - p
    return p, u


all_rows, all_losses = [], []
by_kappa = {k: [] for k in [0, 0.5, 1]}
checks, max_gain, max_gap = 0, 0.0, 0.0
min_p, min_u = float('inf'), float('inf')
for k, kp, kr in product([0, 0.5, 1], [3, 6, 9], [3, 6, 9]):
    a, L = loads(k)
    C = np.array([kp, kp, kr, kr])
    good = np.flatnonzero(np.all(L <= C, axis=1))
    half = good[HALF[good]]
    firm = good[FIRM[good]]
    blind = np.flatnonzero(np.all(L[:, :2] <= kp, axis=1))
    cell = []
    for r, theta in enumerate(costs):
        w = VALUE - Q @ theta
        jf, jh, jn, jb = [best(x, w) for x in
                          [good, half, firm, blind]]
        g = np.full(256, -np.inf)
        np.maximum.at(g, MASK[good], w[good])
        order = sorted(range(8), key=lambda i: (theta[i], i))
        seq = []
        for queue in [range(8), order]:
            mask = 0
            for i in queue:
                trial = mask | (1 << i)
                if g[trial] >= g[mask]:
                    mask = trial
            seq.append(best(good[MASK[good] == mask], w))
        jarrival, jcost = seq
        p, u = pay_util(jf, w, good, theta)
        loss = 100 * (w[jf] - w[[jh, jn, jarrival, jcost]]) / w[jf]
        row = [w[jf], w[jh], w[jn], w[jarrival], w[jcost],
               w[jb], int(np.any(L[jb] > C)),
               np.maximum(L[jb] - C, 0).max(),
               AD[jf].sum(), AD[jn].sum(), Q[jf].sum(),
               p.sum(), u.sum(), *loss]
        all_rows.append(row)
        all_losses.append(loss)
        cell.append(row)
        by_kappa[k].append(row)
        min_p, min_u = min(min_p, p.min()), min(min_u, u.min())
        if (k, kp, kr, r) == (0.5, 6, 6, 0):
            print('WORKED', theta, M[jf], Q[jf], p, u, L[jf])
            print('FEASIBLE OUTCOMES', len(good))
        if r < 4:
            for ids in [good, half, firm]:
                j = best(ids, w)
                p0, u0 = pay_util(j, w, ids, theta)
                for i in range(8):
                    reports = product(range(5), [0, 10, 20, V[i]])
                    for c, b in reports:
                        wr = w + (theta[i] - c) * Q[:, i]
                        wr = wr + (b - V[i]) * AD[:, i]
                        jr = best(ids, wr)
                        absent = ids[M[ids, i] == 0]
                        wm = wr[best(absent, wr)]
                        own_r = b * AD[jr, i] - c * Q[jr, i]
                        pr = wm - (wr[jr] - own_r)
                        true_u = (V[i] * AD[jr, i]
                                  - theta[i] * Q[jr, i] - pr)
                        max_gain = max(max_gain, true_u - u0[i])
                        checks += 1
        if r == 0:
            A = a[:, 1:, :].reshape(24, 4).T
            E = np.kron(np.eye(8), np.ones((1, 3)))
            q = np.array([[0, d / 2, d] for d in D]).ravel()
            coef = -(np.repeat(V, 3) - np.repeat(theta, 3) * q)
            sol = milp(coef, integrality=np.ones(24),
                       bounds=Bounds(0, 1), constraints=[
                           LinearConstraint(E, 0, 1),
                           LinearConstraint(A, -np.inf, C)],
                       options={'mip_rel_gap': 0})
            assert sol.success
            max_gap = max(max_gap, abs(-sol.fun - w[jf]))
    print('CELL', k, kp, kr, np.mean(cell, axis=0))
print('MEAN', np.mean(all_rows, axis=0))
print('MAX LOSSES', np.max(all_losses, axis=0))
for k in by_kappa:
    print('CONCENTRATION', k, np.mean(by_kappa[k], axis=0))
print('AUDIT', checks, max_gain, min_p, min_u, max_gap)

for scale in [0.5, 1, 2]:
    results = []
    for k, kp, kr in product([0, 0.5, 1], [3, 6, 9], [3, 6, 9]):
        a, L = loads(k)
        C = np.array([kp, kp, kr, kr])
        good = np.flatnonzero(np.all(L <= C, axis=1))
        half = good[HALF[good]]
        blind = np.flatnonzero(np.all(L[:, :2] <= kp, axis=1))
        for theta in costs:
            w = VALUE - Q @ (scale * theta)
            j, h, b = [best(x, w) for x in [good, half, blind]]
            results.append([w[j], 100 * (w[j] - w[h]) / w[j],
                            int(np.any(L[b] > C)), AD[j].sum()])
    print('SCALE', scale, np.mean(results, axis=0))
