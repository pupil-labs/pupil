// Geometric Tools, LLC
// Copyright (c) 1998-2014
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.3 (2013/01/03)
//
// Modified by Lech Swirski 2013

#ifndef DistancePointEllipse_h__
#define DistancePointEllipse_h__

#include <Eigen/Core>
#include "geometry/Ellipse.h"

namespace singleeyefitter {

    //----------------------------------------------------------------------------
    // The ellipse is (x0/a)^2 + (x1/b)^2 = 1 with a >= b. The query point is
    // (p0,p1) with p0 >= 0 and p1 >= 0. The function returns the distance from
    // the query point to the ellipse. It also computes the ellipse point (x0,x1)
    // in the first quadrant that is closest to (p0,p1).
    //----------------------------------------------------------------------------
    template <class Real, class Array>
    Real DistancePointEllipseSpecial(Real a, Real b, const Array& p, Eigen::Matrix<Real, 2, 1>& x)
    {
        Real distance;

        if (p.y() > Real(0)) {
            if (p.x() > Real(0)) {
                // Bisect to compute the root of F(t) for t >= -e1*e1.
                Eigen::Array<Real, 2, 1> esqr(a * a, b * b);
                Eigen::Array<Real, 2, 1> ep(a * p.x(), b * p.y());
                Real t0 = -esqr.y() + ep.y();
                Real t1 = -esqr.y() + ep.matrix().norm();
                Real t = t0;
                const int imax = 2 * std::numeric_limits<Real>::max_exponent;

                for (int i = 0; i < imax; ++i) {
                    t = Real(0.5) * (t0 + t1);

                    if (t == t0 || t == t1) {
                        break;
                    }

                    Real r[2] = { ep.x() / (t + esqr[0]), ep.y() / (t + esqr[1]) };
                    Real f = r[0] * r[0] + r[1] * r[1] - Real(1);

                    if (f > Real(0)) {
                        t0 = t;

                    } else if (f < Real(0)) {
                        t1 = t;

                    } else {
                        break;
                    }
                }

                x = esqr * p / (t + esqr);
                distance = (x - p.matrix()).norm();

            } else { // y0 == 0
                x[0] = (Real) 0;
                x[1] = b;
                distance = fabs(p.y() - b);
            }

        } else { // y1 == 0
            Real denom0 = a * a - b * b;
            Real e0y0 = a * p.x();

            if (e0y0 < denom0) {
                // y0 is inside the subinterval.
                Real x0de0 = e0y0 / denom0;
                Real x0de0sqr = x0de0 * x0de0;
                x[0] = a * x0de0;
                x[1] = b * sqrt(fabs(Real(1) - x0de0sqr));
                Real d0 = x[0] - p.x();
                distance = sqrt(d0 * d0 + x[1] * x[1]);

            } else {
                // y0 is outside the subinterval. The closest ellipse point has
                // x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
                x[0] = a;
                x[1] = Real(0);
                distance = fabs(p.x() - a);
            }
        }

        return distance;
    }
    //----------------------------------------------------------------------------
    // The ellipse is (x0/e0)^2 + (x1/e1)^2 = 1. The query point is (y0,y1).
    // The function returns the distance from the query point to the ellipse.
    // It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
    //----------------------------------------------------------------------------
    template <typename Real>
    Real DistancePointEllipse(const Real e[2], const Real y[2], Real x[2])
    {
        // Determine reflections for y to the first quadrant.
        bool reflect[2];
        int i, j;

        for (i = 0; i < 2; ++i) {
            reflect[i] = (y[i] < (Real) 0);
        }

        // Determine the axis order for decreasing extents.
        int permute[2];

        if (e[0] < e[1]) {
            permute[0] = 1; permute[1] = 0;

        } else {
            permute[0] = 0; permute[1] = 1;
        }

        int invpermute[2];

        for (i = 0; i < 2; ++i) {
            invpermute[permute[i]] = i;
        }

        Real locE[2], locY[2];

        for (i = 0; i < 2; ++i) {
            j = permute[i];
            locE[i] = e[j];
            locY[i] = y[j];

            if (reflect[j]) {
                locY[i] = -locY[i];
            }
        }

        Real locX[2];
        Real distance = DistancePointEllipseSpecial(locE, locY, locX);

        // Restore the axis order and reflections.
        for (i = 0; i < 2; ++i) {
            j = invpermute[i];

            if (reflect[j]) {
                locX[j] = -locX[j];
            }

            x[i] = locX[j];
        }

        return distance;
    }
    //----------------------------------------------------------------------------

    template <typename Scalar>
    Scalar DistancePointEllipse(const singleeyefitter::Ellipse2D<Scalar>& ellipse, Scalar x, Scalar y)
    {
        Eigen::Matrix<Scalar, 2, 2> A;
        A << cos(ellipse.angle), sin(ellipse.angle),
        -sin(ellipse.angle), cos(ellipse.angle);
        Eigen::Matrix<Scalar, 2, 1> p(x - ellipse.center.x(), y - ellipse.center.y());
        Eigen::Matrix<Scalar, 2, 1> Ap = A * p;
        // Flip signs to make sure Ap is in the positive quadrant
        Eigen::Matrix<Scalar, 2, 1> Ap_pos = Ap;

        for (int i = 0; i < 2; ++i) {
            if (Ap[i] < 0) { Ap_pos[i] = -Ap_pos[i]; }
        }

        assert(ellipse.major_radius > ellipse.minor_radius);
        Eigen::Matrix<Scalar, 2, 1> el_x;
        auto distance = DistancePointEllipseSpecial(ellipse.major_radius, ellipse.minor_radius, Ap_pos.array(), el_x);

        // Flip signs back
        for (int i = 0; i < 2; ++i) {
            if (Ap[i] < 0) { el_x[i] = -el_x[i]; }
        }

        return distance;
    }

}

#endif // DistancePointEllipse_h__
