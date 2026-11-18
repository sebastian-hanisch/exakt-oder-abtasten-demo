"""Testhilfe: die Gate-Kette in Brüchen (`fractions`), ein vom Code der App unabhängiger Rechenweg (Gauß-Verfahren statt numpy, Matrixprodukt von Hand).

Alles hier ist exakt: π, P^t, d(t), t_mix, σ² und die ESS je Schritt kommen als Brüche heraus. Die Tests vergleichen den numpy-Kern der App (Gleitkomma) damit."""

from __future__ import annotations

from fractions import Fraction as Fr


def matmul(a, b):
    n, m, p = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def identity(n):
    return [[Fr(int(i == j)) for j in range(n)] for i in range(n)]


def inverse(m):
    """Gauß-Jordan in Brüchen."""
    n = len(m)
    a = [list(row) + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col] != 0)
        a[col], a[piv] = a[piv], a[col]
        pv = a[col][col]
        a[col] = [x / pv for x in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [row[n:] for row in a]


class FractionGate:
    """Gate-Kette c Spuren, K Plätze insgesamt, Angebot a (Bruch), uniformisiert mit Λ = größte Austrittsrate."""

    def __init__(self, c: int, k: int, a):
        self.c, self.k, self.a = c, k, Fr(a)
        self.n = k + 1
        up = [self.a if i < k else Fr(0) for i in range(self.n)]
        dn = [Fr(min(i, c)) for i in range(self.n)]
        lam = max(u + d for u, d in zip(up, dn))        # größte Austrittsrate (gleich a + c, solange K > c)
        self.p = [[Fr(0)] * self.n for _ in range(self.n)]
        for i in range(self.n):
            if i < k:
                self.p[i][i + 1] = up[i] / lam
            if i > 0:
                self.p[i][i - 1] = dn[i] / lam
            self.p[i][i] = 1 - sum(self.p[i][j] for j in range(self.n) if j != i)
        w = [Fr(1)]
        for i in range(1, self.n):
            w.append(w[-1] * self.a / min(i, c))
        self.pi = [x / sum(w) for x in w]

    def pi_by_linear_system(self):
        """π aus dem linearen System π(P − I) = 0 mit einer durch die Normierung ersetzten Gleichung (zweiter Rechenweg in Brüchen)."""
        a_mat = [[self.p[i][j] - (1 if i == j else 0) for i in range(self.n)] for j in range(self.n)]
        a_mat[0] = [Fr(1)] * self.n
        inv = inverse(a_mat)
        return [inv[i][0] for i in range(self.n)]

    def d_curve(self, t_max: int):
        pt = identity(self.n)
        out = []
        for _ in range(t_max + 1):
            out.append(max(sum(abs(pt[i][j] - self.pi[j]) for j in range(self.n)) / 2 for i in range(self.n)))
            pt = matmul(pt, self.p)
        return out

    def t_mix(self, eps=Fr(1, 4), t_max: int = 200):
        for t, x in enumerate(self.d_curve(t_max)):
            if x <= eps:
                return t
        return None

    def sigma2(self, f):
        """σ² = 2⟨f~, Z f~⟩_π − ⟨f~, f~⟩_π, Z = (I − P + 1π)⁻¹ (Brüche); gibt (σ², Var_π(f)) zurück."""
        mz = [[Fr(int(i == j)) - self.p[i][j] + self.pi[j] for j in range(self.n)] for i in range(self.n)]
        z = inverse(mz)
        mu = sum(self.pi[i] * f[i] for i in range(self.n))
        ft = [x - mu for x in f]
        zf = [sum(z[i][j] * ft[j] for j in range(self.n)) for i in range(self.n)]
        var = sum(self.pi[i] * ft[i] ** 2 for i in range(self.n))
        return 2 * sum(self.pi[i] * ft[i] * zf[i] for i in range(self.n)) - var, var
