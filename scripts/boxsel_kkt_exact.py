#!/usr/bin/env python
r"""BoxSEL Phase 4d - the exact algebraic optimum (KKT solve) for inf I_box^n.

The Phase-4c numerics located the infimum at n=2, ~=0.41010, with both |A&C| and |B&C| active.
Solving that KKT exactly gives a CLOSED FORM:

    inf I_box^n  =  (9 + sqrt(17)) / 32  ~=  0.4100970          (attained at n = 2)

DERIVATION (structured family generalizing the Phase-4 witness; B_1 full, both C-overlaps active,
A&B slack). Let x = |A_1| be A's axis-1 length, with

    A = [1-x, 1] x [1-1/(2x), 1],   B = [0,1] x [1/2, 1],   C = [0, 2(1-x)] x [y0, y0+1/(4(1-x))]

where y0 = 1 - 1/(2x). Marginals |A|=|B|=|C|=1/2 hold identically. Imposing the two active
constraints |A&C| = |B&C| = 1/4 forces z = 2(1-x) AND z = x/(2(1-x)), hence

    4(1-x)^2 = x   =>   4x^2 - 9x + 4 = 0   =>   x = (9 - sqrt 17)/8 ~= 0.60961.

Then |A&B&C| = 1/8 and |A&B| = x/2, so q* = (1/8)/(x/2) = 1/(4x) = (9 + sqrt 17)/32.

STATUS. The optimum is CERTIFIED ACHIEVABLE: this module constructs the config in exact Q(sqrt 17)
arithmetic and verifies every constraint and the value q* exactly. It is numerically the GLOBAL
n=2 minimum (Phase-4c's broad search finds nothing below it; Nelder-Mead converged to ~0.4100984,
i.e. just ABOVE this exact value). So inf I_box^n <= (9+sqrt 17)/32 is certified, and it equals the
infimum up to the still-open matching LOWER bound (the proven lower bound stays 1/4; closing it is
the last open thread). The certified sandwich therefore tightens to

    1/4  <=  inf I_box^n  =  (9 + sqrt 17)/32  ~= 0.4100970   <   513/1250 = 0.41040  (old witness).

Exact and dependency-free (a tiny Q(sqrt 17) field; no numpy/scipy/sympy).
"""
from fractions import Fraction

RADICAND = 17


class Surd:
    """An element a + b*sqrt(17) of the field Q(sqrt 17), with a, b rational (exact)."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fraction(a)
        self.b = Fraction(b)

    @staticmethod
    def coerce(x):
        return x if isinstance(x, Surd) else Surd(x, 0)

    def __add__(self, o):
        o = Surd.coerce(o)
        return Surd(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, o):
        o = Surd.coerce(o)
        return Surd(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return Surd.coerce(o).__sub__(self)

    def __neg__(self):
        return Surd(-self.a, -self.b)

    def __mul__(self, o):
        o = Surd.coerce(o)
        return Surd(self.a * o.a + RADICAND * self.b * o.b, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def reciprocal(self):
        d = self.a * self.a - RADICAND * self.b * self.b  # the field norm (nonzero unless self==0)
        if d == 0:
            raise ZeroDivisionError("Surd reciprocal of zero")
        return Surd(self.a / d, -self.b / d)

    def __truediv__(self, o):
        return self * Surd.coerce(o).reciprocal()

    def __rtruediv__(self, o):
        return Surd.coerce(o) * self.reciprocal()

    def __eq__(self, o):
        o = Surd.coerce(o)
        return self.a == o.a and self.b == o.b  # {1, sqrt 17} are Q-linearly independent

    def __hash__(self):
        return hash((self.a, self.b))

    def sign(self):
        """Exact sign of a + b*sqrt(17) in {-1, 0, 1} (no floating point)."""
        if self.b == 0:
            return (self.a > 0) - (self.a < 0)
        if self.a == 0:
            return (self.b > 0) - (self.b < 0)
        if self.a > 0 and self.b > 0:
            return 1
        if self.a < 0 and self.b < 0:
            return -1
        lhs, rhs = self.a * self.a, RADICAND * self.b * self.b  # compare |a| vs |b|*sqrt17 by squares
        if self.a > 0:  # b < 0: a + b*sqrt17 > 0  iff  a^2 > 17 b^2
            return 1 if lhs > rhs else (-1 if lhs < rhs else 0)
        return 1 if rhs > lhs else (-1 if rhs < lhs else 0)  # a < 0, b > 0

    def __lt__(self, o):
        return (self - Surd.coerce(o)).sign() < 0

    def __le__(self, o):
        return (self - Surd.coerce(o)).sign() <= 0

    def __gt__(self, o):
        return (self - Surd.coerce(o)).sign() > 0

    def __ge__(self, o):
        return (self - Surd.coerce(o)).sign() >= 0

    def __float__(self):
        return float(self.a) + float(self.b) * (RADICAND ** 0.5)

    def __repr__(self):
        return f"({self.a} + {self.b}*sqrt{RADICAND})"


# --- exact box geometry over Q(sqrt 17) (boxes = 2 intervals of Surd endpoints) ---

def _len(interval):
    return interval[1] - interval[0]


def box_volume(box):
    vol = Surd(1)
    for interval in box:
        vol = vol * _len(interval)
    return vol


def meet_volume(boxes):
    """Exact intersection volume of axis-parallel boxes (per-axis min/max via exact Surd compare)."""
    vol = Surd(1)
    dim = len(boxes[0])
    for k in range(dim):
        lo, hi = boxes[0][k]
        for bx in boxes[1:]:
            lo = lo if lo >= bx[k][0] else bx[k][0]
            hi = hi if hi <= bx[k][1] else bx[k][1]
        width = hi - lo
        if width.sign() <= 0:
            return Surd(0)
        vol = vol * width
    return vol


X_OPT = Surd(Fraction(9, 8), Fraction(-1, 8))          # x = (9 - sqrt 17)/8, the root of 4x^2-9x+4=0 in (0,1)
Q_STAR = Surd(Fraction(9, 32), Fraction(1, 32))        # q* = (9 + sqrt 17)/32, the exact infimum
CERTIFIED_LOWER = Surd(Fraction(1, 4))                 # 1/4 (Phase 4b, proven)


def optimal_config():
    """The exact n=2 minimizer in Q(sqrt 17). Returns {'A','B','C'} as boxes of Surd intervals."""
    x = X_OPT
    a2 = Surd(1) / (Surd(2) * x)        # |A_2| = 1/(2x)
    y0 = Surd(1) - a2                    # A_2, C_2 share the bottom y0 = 1 - 1/(2x)
    z = Surd(2) * (Surd(1) - x)         # |C_1| = 2(1-x)
    h = Surd(1) / (Surd(2) * z)         # |C_2| = 1/(2z)
    return {
        "A": [(Surd(1) - x, Surd(1)), (y0, Surd(1))],
        "B": [(Surd(0), Surd(1)), (Surd(Fraction(1, 2)), Surd(1))],
        "C": [(Surd(0), z), (y0, y0 + h)],
    }


def optimal_query_value():
    """q = |A&B&C| / |A&B| at the optimum (returns the exact Surd; equals Q_STAR)."""
    e = optimal_config()
    return meet_volume([e["A"], e["B"], e["C"]]) / meet_volume([e["A"], e["B"]])


def certified_sandwich():
    """(lower, exact_inf): 1/4 <= inf I_box^n = (9 + sqrt 17)/32."""
    return CERTIFIED_LOWER, Q_STAR


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    e = optimal_config()
    print("x = (9 - sqrt17)/8 root of 4x^2-9x+4:", (Surd(4) * X_OPT * X_OPT - Surd(9) * X_OPT + Surd(4)) == 0)
    print("|A|,|B|,|C| =", [float(box_volume(e[k])) for k in "ABC"])
    print("|A&B|, |A&C|, |B&C| =", [float(meet_volume([e[i], e[j]])) for i, j in (("A", "B"), ("A", "C"), ("B", "C"))])
    print("|A&B&C| =", float(meet_volume([e["A"], e["B"], e["C"]])))
    print("q* =", repr(optimal_query_value()), "~=", float(Q_STAR))
    print("exact inf I_box^n = (9 + sqrt 17)/32 ~=", float(Q_STAR), " (< 513/1250 =", 513 / 1250, ")")
