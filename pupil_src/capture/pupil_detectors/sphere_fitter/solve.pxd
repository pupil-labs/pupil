"""
    Andrew Xia playing around with porting c++ code to python
    This file was originally located in geometry/solve.py, now in the parent folder just as solve.py
    HUDING
    Created July 2 2015

    The inputs to the solve equation are tuples a,b,c

"""
cimport libc.math as cmath
import cython
@cython.cdivision(True)
cdef inline double solve(double a):
    if (a == 0):
        return 0
    else:
        raise ValueError("No Solution")

@cython.cdivision(True)

cdef inline double solve_two(double a,double b):
    if (a == 0):
        return solve(b)
    return -b/a

@cython.cdivision(True)
cdef inline solve_three(double a,double b,double c):
    if (a == 0):
        root = solve_two(b,c)
        return root,root

    # http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
    # Pg 184
    det = cmath.pow(b,2) - 4*a*c
    if (det < 0):
        raise ValueError("No Solution")
    sqrtdet = cmath.sqrt(det)
    if b >= 0:
        q = -0.5*(b + cmath.sqrt(det))
    else:
        q = -0.5*(b - cmath.sqrt(det))
    return q/a, q/c

@cython.cdivision(True)
cdef inline solve_four(double a,double b,double c,double d):
    if (a == 0):
        roots = solve_three(b,c,d)
        return roots[0],roots[1],roots[1]

    # http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
    # http://web.archive.org/web/20120321013251/http://linus.it.uts.edu.au/~don/pubs/solving.html
    p = b/a
    q = c/a
    r = d/a
    u = q - cmath.pow(p,2)/3
    v = r - p*q/3 + 2*p*p*p/27
    j = 4*u*u*u/27 * v*v

    if (b == 0 and c == 0):
        return cmath.cbrt(-d),cmath.cbrt(-d),cmath.cbrt(-d)
    elif (abs(p) > 10e100):
        return -p,-p,-p
    elif (abs(q) > 10e100):
        return -cmath.cbrt(v),-cmath.cbrt(v),-cmath.cbrt(v)
    elif (abs(u) > 10e100): #some big number
        return cmath.cbrt(4)*u/3,cmath.cbrt(4)*u/3,cmath.cbrt(4)*u/3

    if (j > 0):
        #one real root
        w = cmath.sqrt(j)
        if (v > 0):
            y = (u / 3)*cmath.cbrt(2 / (w + v)) - cmath.cbrt((w + v) / 2) - p / 3;
        else:
            y = cmath.cbrt((w - v) / 2) - (u / 3)*cmath.cbrt(2 / (w - v)) - p / 3;
        return y,y,y
    else:
        s = cmath.sqrt(-u/3)
        t = -v/ (2*s*s*s)
        k = cmath.acos(t)/3

        y1 = 2 * s*cmath.cos(k) - p / 3;
        y2 = s*(-cmath.cos(k) + cmath.sqrt(3.)*cmath.sin(k)) - p / 3;
        y3 = s*(-cmath.cos(k) - cmath.sqrt(3.)*cmath.sin(k)) - p / 3;
        return y1,y2,y3

