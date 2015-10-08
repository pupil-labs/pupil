#ifndef _CONICOID_H_
#define _CONICOID_H_

#include <Eigen/Core>

namespace singleeyefitter {

    template<typename T>
    class Conic;

    // Conicoid (quartic surface) of the form:
    // Ax^2 + By^2 + Cz^2 + 2Fyz + 2Gzx + 2Hxy + 2Ux + 2Vy + 2Wz + D = 0
    template<typename T>
    class Conicoid {
        public:
            typedef T Scalar;

            Scalar A, B, C, F, G, H, U, V, W, D;

            Conicoid(Scalar A, Scalar B, Scalar C, Scalar F, Scalar G, Scalar H, Scalar D)
                : A(A), B(B), C(C), F(F), G(G), H(H), U(U), V(V), W(W), D(D)
            {
            }

            template<typename ConicScalar, typename Derived>
            explicit Conicoid(const Conic<ConicScalar>& conic, const Eigen::MatrixBase<Derived>& vertex)
            {
                static_assert(Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == 3, "Cone vertex requires 3 element vector as vector type");
                using math::sq;
                // Finds conicoid with given conic base and vertex
                // Assumes conic is on the plane z = 0
                auto alpha = vertex[0];
                auto beta = vertex[1];
                auto gamma = vertex[2];
                A = sq(gamma) * conic.A;
                B = sq(gamma) * conic.C;
                C = conic.A * sq(alpha) + conic.B * alpha * beta + conic.C * sq(beta) + conic.D * alpha + conic.E * beta + conic.F;
                F = -gamma * (conic.C * beta + conic.B / 2 * alpha + conic.E / 2);
                G = -gamma * (conic.B / 2 * beta + conic.A * alpha + conic.D / 2);
                H = sq(gamma) * conic.B / 2;
                U = sq(gamma) * conic.D / 2;
                V = sq(gamma) * conic.E / 2;
                W = -gamma * (conic.E / 2 * beta + conic.D / 2 * alpha + conic.F);
                D = sq(gamma) * conic.F;
            }

            Scalar operator()(Scalar x, Scalar y, Scalar z) const
            {
                return A * sq(x) + B * sq(y) + C * sq(z) + 2 * F * y * z + 2 * G * x * z + 2 * H * x * y + 2 * U * x + 2 * V * y + 2 * W * z + D;
            }

            Conic<Scalar> intersectZ(Scalar z = Scalar(0)) const
            {
                // Finds conic at given z intersection
                // Ax^2 + By^2 + Cz^2 + 2Fyz + 2Gzx + 2Hxy + 2Ux + 2Vy + 2Wz + D = 0
                // becomes
                // Ax^2 + Bxy + Cy^2 + Fx + Ey + D = 0
                return Conic<Scalar>(A,
                                     2 * H,
                                     B,
                                     2 * G * z + 2 * U,
                                     2 * F * z + 2 * V,
                                     C * sq(z) + 2 * W * z + D);
            }

            template<typename ADerived, typename TDerived>
            Conicoid<Scalar> transformed(const Eigen::MatrixBase<ADerived>& a, const Eigen::MatrixBase<TDerived>& t) const
            {
                static_assert(ADerived::RowsAtCompileTime == 3 && ADerived::ColsAtCompileTime == 3, "Affine transform must be 3x3 matrix");
                static_assert(TDerived::IsVectorAtCompileTime && TDerived::SizeAtCompileTime == 3, "Translation must be 3 element vector");
                // We map x,y,z to a new space using
                //     [x y z] -> affine*[x y z] + translation
                //
                // Using a for affine and t for translation:
                //     x -> a_00*x + a01*y + a02*z + t0
                //     y -> a_10*x + a11*y + a12*z + t1
                //     z -> a_20*x + a21*y + a22*z + t2
                //
                // So
                //     Ax^2 + By^2 + Cz^2 + 2Fyz + 2Gzx + 2Hxy + 2Ux + 2Vy + 2Wz + D
                // becomes
                //       A(a_00*x + a01*y + a02*z + t0)(a_00*x + a01*y + a02*z + t0)
                //     + B(a_10*x + a11*y + a12*z + t1)(a_10*x + a11*y + a12*z + t1)
                //     + C(a_20*x + a21*y + a22*z + t2)(a_20*x + a21*y + a22*z + t2)
                //     + 2F(a_10*x + a11*y + a12*z + t1)(a_20*x + a21*y + a22*z + t2)
                //     + 2G(a_20*x + a21*y + a22*z + t2)(a_00*x + a01*y + a02*z + t0)
                //     + 2H(a_00*x + a01*y + a02*z + t0)(a_10*x + a11*y + a12*z + t1)
                //     + 2U(a_00*x + a01*y + a02*z + t0)
                //     + 2V(a_10*x + a11*y + a12*z + t1)
                //     + 2W(a_20*x + a21*y + a22*z + t2)
                //     + D
                //
                // Collecting terms gives:
                return Conicoid<Scalar>(
                           A * sq(a(0, 0)) + B * sq(a(1, 0)) + C * sq(a(2, 0)) + Scalar(2) * F * a(1, 0) * a(2, 0) + Scalar(2) * G * a(0, 0) * a(2, 0) + Scalar(2) * H * a(0, 0) * a(1, 0),
                           A * sq(a(0, 1)) + B * sq(a(1, 1)) + C * sq(a(2, 1)) + Scalar(2) * F * a(1, 1) * a(2, 1) + Scalar(2) * G * a(0, 1) * a(2, 1) + Scalar(2) * H * a(0, 1) * a(1, 1),
                           A * sq(a(0, 2)) + B * sq(a(1, 2)) + C * sq(a(2, 2)) + Scalar(2) * F * a(1, 2) * a(2, 2) + Scalar(2) * G * a(0, 2) * a(2, 2) + Scalar(2) * H * a(0, 2) * a(1, 2),
                           A * a(0, 1) * a(0, 2) + B * a(1, 1) * a(1, 2) + C * a(2, 1) * a(2, 2) + F * a(1, 1) * a(2, 2) + F * a(1, 2) * a(2, 1) + G * a(0, 1) * a(2, 2) + G * a(0, 2) * a(2, 1) + H * a(0, 1) * a(1, 2) + H * a(0, 2) * a(1, 1),
                           A * a(0, 2) * a(0, 0) + B * a(1, 2) * a(1, 0) + C * a(2, 2) * a(2, 0) + F * a(1, 2) * a(2, 0) + F * a(1, 0) * a(2, 2) + G * a(0, 2) * a(2, 0) + G * a(0, 0) * a(2, 2) + H * a(0, 2) * a(1, 0) + H * a(0, 0) * a(1, 2),
                           A * a(0, 0) * a(0, 1) + B * a(1, 0) * a(1, 1) + C * a(2, 0) * a(2, 1) + F * a(1, 0) * a(2, 1) + F * a(1, 1) * a(2, 0) + G * a(0, 0) * a(2, 1) + G * a(0, 1) * a(2, 0) + H * a(0, 0) * a(1, 1) + H * a(0, 1) * a(1, 0),
                           A * a(0, 0) * t(0) + B * a(1, 0) * t(1) + C * a(2, 0) * t(2) + F * a(1, 0) * t(2) + F * a(2, 0) * t(1) + G * a(0, 0) * t(2) + G * a(2, 0) * t(0) + H * a(0, 0) * t(1) + H * a(1, 0) * t(0) + U * a(0, 0) + V * a(1, 0) + W * a(2, 0),
                           A * a(0, 1) * t(0) + B * a(1, 1) * t(1) + C * a(2, 1) * t(2) + F * a(1, 1) * t(2) + F * a(2, 1) * t(1) + G * a(0, 1) * t(2) + G * a(2, 1) * t(0) + H * a(0, 1) * t(1) + H * a(1, 1) * t(0) + U * a(0, 1) + V * a(1, 1) + W * a(2, 1),
                           A * a(0, 2) * t(0) + B * a(1, 2) * t(1) + C * a(2, 2) * t(2) + F * a(1, 2) * t(2) + F * a(2, 2) * t(1) + G * a(0, 2) * t(2) + G * a(2, 2) * t(0) + H * a(0, 2) * t(1) + H * a(1, 2) * t(0) + U * a(0, 2) + V * a(1, 2) + W * a(2, 2),
                           A * sq(t(0)) + B * sq(t(1)) + C * sq(t(2)) + Scalar(2) * F * t(1) * t(2) + Scalar(2) * G * t(0) * t(2) + Scalar(2) * H * t(0) * t(1) + Scalar(2) * U * t(0) + Scalar(2) * V * t(1) + Scalar(2) * W * t(2) + D
                       );
            }
    };

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Conicoid<T>& conicoid)
    {
        return os << "Conicoid { " << conicoid.A << "x^2 + " << conicoid.B << "y^2 + " << conicoid.C << "z^2 + "
               "2*" << 2 * conicoid.F << "yz + 2*" << 2 * conicoid.G << "zx + 2*" << 2 * conicoid.H << "xy + "
               "2*" << 2 * conicoid.U << "x + 2*" << 2 * conicoid.V << "y + 2*" << 2 * conicoid.W << "z + " << conicoid.D << " = 0 }";
    }

}

#endif
