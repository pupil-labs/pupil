#ifndef _CONIC_H_
#define _CONIC_H_

#include <Eigen/Core>
#include "mathHelper.h"

namespace singleeyefitter {

    template<typename T>
    class Ellipse2D;

    template<typename T>
    class Conic {
        public:
            typedef T Scalar;

            Scalar A, B, C, D, E, F;

            Conic(Scalar A, Scalar B, Scalar C, Scalar D, Scalar E, Scalar F)
                : A(A), B(B), C(C), D(D), E(E), F(F)
            {
            }

            template<typename U>
            explicit Conic(const Ellipse2D<U>& ellipse)
            {
                using std::sin;
                using std::cos;
                using singleeyefitter::math::sq;
                auto ax = cos(ellipse.angle);
                auto ay = sin(ellipse.angle);
                auto a2 = sq(ellipse.major_radius);
                auto b2 = sq(ellipse.minor_radius);
                A = ax * ax / a2 + ay * ay / b2;
                B = 2 * ax * ay / a2 - 2 * ax * ay / b2;
                C = ay * ay / a2 + ax * ax / b2;
                D = (-2 * ax * ay * ellipse.center[1] - 2 * ax * ax * ellipse.center[0]) / a2
                    + (2 * ax * ay * ellipse.center[1] - 2 * ay * ay * ellipse.center[0]) / b2;
                E = (-2 * ax * ay * ellipse.center[0] - 2 * ay * ay * ellipse.center[1]) / a2
                    + (2 * ax * ay * ellipse.center[0] - 2 * ax * ax * ellipse.center[1]) / b2;
                F = (2 * ax * ay * ellipse.center[0] * ellipse.center[1] + ax * ax * ellipse.center[0] * ellipse.center[0] + ay * ay * ellipse.center[1] * ellipse.center[1]) / a2
                    + (-2 * ax * ay * ellipse.center[0] * ellipse.center[1] + ay * ay * ellipse.center[0] * ellipse.center[0] + ax * ax * ellipse.center[1] * ellipse.center[1]) / b2
                    - 1;
            }

            Scalar operator()(Scalar x, Scalar y) const
            {
                return A * x * x + B * x * y + C * y * y + D * x + E * y + F;
            }

            template<typename ADerived, typename TDerived>
            Conic<Scalar> transformed(const Eigen::MatrixBase<ADerived>& a, const Eigen::MatrixBase<TDerived>& t) const
            {
                static_assert(ADerived::RowsAtCompileTime == 2 && ADerived::ColsAtCompileTime == 2, "Affine transform must be 2x2 matrix");
                static_assert(TDerived::IsVectorAtCompileTime && TDerived::SizeAtCompileTime == 2, "Translation must be 2 element vector");
                // We map x,y to a new space using
                //     [x y] -> affine*[x y] + translation
                //
                // Using a for affine and t for translation:
                //     x -> a_00*x + a01*y + t0
                //     y -> a_10*x + a11*y + t1
                //
                // So
                //     Ax^2 + Bxy + Cy^2 + Dx + Ey + F
                // becomes
                //       A(a_00*x + a01*y + t0)(a_00*x + a01*y + t0)
                //     + B(a_00*x + a01*y + t0)(a_10*x + a11*y + t1)
                //     + C(a_10*x + a11*y + t1)(a_10*x + a11*y + t1)
                //     + D(a_00*x + a01*y + t0)
                //     + E(a_10*x + a11*y + t1)
                //     + F
                //
                // Collecting terms gives:
                return Conic<Scalar>(
                           A * sq(a(0, 0)) + B * a(0, 0) * a(1, 0) + C * sq(a(1, 0)),
                           2 * A * a(0, 0) * a(0, 1) + B * a(0, 0) * a(1, 1) + B * a(0, 1) * a(1, 0) + 2 * C * a(1, 0) * a(1, 1),
                           A * sq(a(0, 1)) + B * a(0, 1) * a(1, 1) + C * sq(a(1, 1)),
                           2 * A * a(0, 0) * t(0) + B * a(0, 0) * t(1) + B * a(1, 0) * t(0) + 2 * C * a(1, 0) * t(1) + D * a(0, 0) + E * a(1, 0),
                           2 * A * a(0, 1) * t(0) + B * a(0, 1) * t(1) + B * a(1, 1) * t(0) + 2 * C * a(1, 1) * t(1) + D * a(0, 1) + E * a(1, 1),
                           A * sq(t(0)) + B * t(0) * t(1) + C * sq(t(1)) + D * t(0) + E * t(1) + F
                       );
            }
    };

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Conic<T>& conic)
    {
        return os << "Conic { " << conic.A << "x^2 + " << conic.B << "xy + " << conic.C << "y^2 + " << conic.D << "x + " << conic.E << "y + " << conic.F << " = 0 } ";
    }

}

#endif
