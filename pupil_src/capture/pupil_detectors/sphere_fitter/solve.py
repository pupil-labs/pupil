"""
    Andrew Xia playing around with porting c++ code to python
    This file was originally located in geometry/solve.py, now in the parent folder just as solve.py
    HUDING
    Created July 2 2015

    The inputs to the solve equation are tuples a,b,c

"""
from math import cos,acos,sin,sqrt,pow
import numpy as np
def cbrt(value):
	return value**(1./3.)
def solve(a):
    if (a == 0):
        return 0
    else:
        raise ValueError("No Solution")


def solve_two(a,b):
    if (a == 0):
        return solve(b)
    return -b/a

def solve_three(a,b,c):
    if (a == 0):
        root = solve_two(b,c)
        return root,root

    # http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
    # Pg 184
    det = pow(b,2) - 4*a*c
    if (det < 0):
        raise ValueError("No Solution")
    sqrtdet = sqrt(det)
    if b >= 0:
        q = -0.5*(b + sqrt(det))
    else:
        q = -0.5*(b - sqrt(det))
    return q/a, q/c

def solve_four(a,b,c,d):
    if (a == 0):
        roots = solve_three(b,c,d)
        return np.asarray([roots[0],roots[1],roots[1]]).reshape(-1)

    # http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
    # http://web.archive.org/web/20120321013251/http://linus.it.uts.edu.au/~don/pubs/solving.html
    p = b/a
    q = c/a
    r = d/a
    u = q - pow(p,2)/3
    v = r - p*q/3 + 2*p*p*p/27
    j = 4*u*u*u/27 * v*v

    if (b == 0 and c == 0):
        return np.asarray([cbrt(-d),cbrt(-d),cbrt(-d)]).reshape(-1)
    elif (abs(p) > 10e100):
        return np.asarray([-p,-p,-p]).reshape(-1)
    elif (abs(q) > 10e100):
        return np.asarray([-cbrt(v),-cbrt(v),-cbrt(v)]).reshape(-1)
    elif (abs(u) > 10e100): #some big number
        return np.asarray([cbrt(4)*u/3,cbrt(4)*u/3,cbrt(4)*u/3]).reshape(-1)

    if (j > 0):
        #one real root
        w = sqrt(j)
        if (v > 0):
            y = (u / 3)*cbrt(2 / (w + v)) - cbrt((w + v) / 2) - p / 3;
        else:
            y = cbrt((w - v) / 2) - (u / 3)*cbrt(2 / (w - v)) - p / 3;
        return np.asarray([y,y,y]).reshape(-1)
    else:
        s = sqrt(-u/3)
        t = -v/ (2*s*s*s)
        k = acos(t)/3

        y1 = 2 * s*cos(k) - p / 3;
        y2 = s*(-cos(k) + sqrt(3.)*sin(k)) - p / 3;
        y3 = s*(-cos(k) - sqrt(3.)*sin(k)) - p / 3;
        return np.asarray([y1,y2,y3]).reshape(-1)
