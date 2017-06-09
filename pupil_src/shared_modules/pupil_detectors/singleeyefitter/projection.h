#ifndef singleeyefitter_project_h__
#define singleeyefitter_project_h__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "geometry/Ellipse.h"
#include "geometry/Circle.h"
#include "geometry/Conic.h"
#include "geometry/Sphere.h"
#include "geometry/Conicoid.h"
#include "mathHelper.h"
#include "solve.h"

namespace singleeyefitter {

    template<typename Scalar>
    Conic<Scalar> project(const Circle3D<Scalar>& circle, Scalar focal_length)
    {
        typedef typename Circle3D<Scalar>::Vector Vector;
        using math::sq;
        Vector c = circle.center;
        Vector n = circle.normal;
        Scalar r = circle.radius;
        Scalar f = focal_length;
        // Construct cone with circle as base and vertex v = (0,0,0).
        //
        // For the circle,
        //     |p - c|^2 = r^2 where (p-c).n = 0 (i.e. on the circle plane)
        //
        // A cone is basically concentric circles, with center on the line c->v.
        // For any point p, the corresponding circle center c' is the intersection
        // of the line c->v and the plane through p normal to n. So,
        //
        //     d = ((p - v).n)/(c.n)
        //     c' = d c + v
        //
        // The radius of these circles decreases linearly as you approach 0, so
        //
        //     |p - c'|^2 = (r*|c' - v|/|c - v|)^2
        //
        // Since v = (0,0,0), this simplifies to
        //
        //     |p - (p.n/c.n)c|^2 = (r*|(p.n/c.n)c|/|c|)^2
        //
        //     |(c.n)p - (p.n)c|^2         / p.n \^2
        //     ------------------- = r^2 * | --- |
        //           (c.n)^2               \ c.n /
        //
        //     |(c.n)p - (p.n)c|^2 - r^2 * (p.n)^2 = 0
        //
        // Expanding out p, c and n gives
        //
        //     |(c.n)x - (x*n_x + y*n_y + z*n_z)c_x|^2
        //     |(c.n)y - (x*n_x + y*n_y + z*n_z)c_y|   - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
        //     |(c.n)z - (x*n_x + y*n_y + z*n_z)c_z|
        //
        //       ((c.n)x - (x*n_x + y*n_y + z*n_z)c_x)^2
        //     + ((c.n)y - (x*n_x + y*n_y + z*n_z)c_y)^2
        //     + ((c.n)z - (x*n_x + y*n_y + z*n_z)c_z)^2
        //     - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
        //
        //       (c.n)^2 x^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*x*c_x + (x*n_x + y*n_y + z*n_z)^2 c_x^2
        //     + (c.n)^2 y^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*y*c_y + (x*n_x + y*n_y + z*n_z)^2 c_y^2
        //     + (c.n)^2 z^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*z*c_z + (x*n_x + y*n_y + z*n_z)^2 c_z^2
        //     - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
        //
        //       (c.n)^2 x^2 - 2*(c.n)*c_x*(x*n_x + y*n_y + z*n_z)*x
        //     + (c.n)^2 y^2 - 2*(c.n)*c_y*(x*n_x + y*n_y + z*n_z)*y
        //     + (c.n)^2 z^2 - 2*(c.n)*c_z*(x*n_x + y*n_y + z*n_z)*z
        //     + (x*n_x + y*n_y + z*n_z)^2 * (c_x^2 + c_y^2 + c_z^2 - r^2)
        //
        //       (c.n)^2 x^2 - 2*(c.n)*c_x*(x*n_x + y*n_y + z*n_z)*x
        //     + (c.n)^2 y^2 - 2*(c.n)*c_y*(x*n_x + y*n_y + z*n_z)*y
        //     + (c.n)^2 z^2 - 2*(c.n)*c_z*(x*n_x + y*n_y + z*n_z)*z
        //     + (|c|^2 - r^2) * (n_x^2*x^2 + n_y^2*y^2 + n_z^2*z^2 + 2*n_x*n_y*x*y + 2*n_x*n_z*x*z + 2*n_y*n_z*y*z)
        //
        // Collecting conicoid terms gives
        //
        //       [xyz]^2 : (c.n)^2 - 2*(c.n)*c_[xyz]*n_[xyz] + (|c|^2 - r^2)*n_[xyz]^2
        //    [yzx][zxy] : - 2*(c.n)*c_[yzx]*n_[zxy] - 2*(c.n)*c_[zxy]*n_[yzx] + (|c|^2 - r^2)*2*n_[yzx]*n_[zxy]
        //               : 2*((|c|^2 - r^2)*n_[yzx]*n_[zxy] - (c,n)*(c_[yzx]*n_[zxy] + c_[zxy]*n_[yzx]))
        //         [xyz] : 0
        //             1 : 0
        Scalar cn = c.dot(n);
        Scalar c2r2 = (c.dot(c) - sq(r));
        Vector ABC = (sq(cn) - 2.0 * cn * c.array() * n.array() + c2r2 * n.array().square());
        Scalar F = 2.0 * (c2r2 * n(1) * n(2) - cn * (n(1) * c(2) + n(2) * c(1)));
        Scalar G = 2.0 * (c2r2 * n(2) * n(0) - cn * (n(2) * c(0) + n(0) * c(2)));
        Scalar H = 2.0 * (c2r2 * n(0) * n(1) - cn * (n(0) * c(1) + n(1) * c(0)));
        // Then set z=f to get conic which is the result of intersecting the cone with the focal plane
        return Conic<Scalar>(
                   ABC(0), // x^2 (Ax^2)
                   H, // xy (Hxy)
                   ABC(1), // y^2 (By^2)
                   G * f /*+ Const(0)*/, // x (Gxz + Ux, z = f)
                   F * f /*+ Const(0)*/, // y (Fyz + Vy, z = f)
                   ABC(2) * sq(f) /*+ Const(0)*f + Const(0)*/ // 1 (Cz^2 + Wz + D, z = f)
               );
    }

    /*template<typename Scalar, typename PDerived, typename NDerived>
    typename Conic<Scalar> project(Conic<Scalar> conic, const Eigen::DenseBase<PDerived>& point, const Eigen::DenseBase<NDerived>& normal, Scalar focal_length)
    {
    // Consider two coordinate systems:
    //    camera (camera at 0, x,y aligned with image plane, z going away from camera)
    //    conic (conic on xy-plane, with plane normal = (0,0,1) and plane point = (0,0,0) )
    //
    // To project conic lying on plane defined by point and normal (point corresponding to (0,0) in conic's 2D space), do:
    //
    //     Input as in camera space,
    //     Transform to conic space,
    //     Form conicoid with conic as base and camera center as vertex
    //     Transform back to camera space
    //     Intersect conicoid with image plane (z=f)

    Eigen::Matrix<Scalar,3,1> camera_center(0,0,0);
    }*/


    template<typename Scalar>
    Ellipse2D<Scalar> project(const Sphere<Scalar>& sphere, Scalar focal_length)
    {
        return Ellipse2D<Scalar>(
                   focal_length * sphere.center.template head<2>() / sphere.center[2],
                   focal_length * sphere.radius / sphere.center[2],
                   focal_length * sphere.radius / sphere.center[2],
                   0);
    }
    template<typename Derived>
    typename Eigen::DenseBase<Derived>::template FixedSegmentReturnType<2>::Type::PlainObject project(const Eigen::DenseBase<Derived>& point, typename Eigen::DenseBase<Derived>::Scalar focal_length)
    {
        static_assert(Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == 3, "Point must be 3 element vector");
        return focal_length * point.template head<2>() / point(2);
    }


    template<typename Scalar>
    std::pair<Circle3D<Scalar>, Circle3D<Scalar>> unproject(const Ellipse2D<Scalar>& ellipse, Scalar circle_radius, Scalar focal_length)
    {
        using std::sqrt;
        using std::abs;
        using math::sign;
        using math::sq;
        typedef Conic<Scalar> Conic;
        typedef Conicoid<Scalar> Conicoid;
        typedef Circle3D<Scalar> Circle;
        typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
        typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
        typedef Eigen::Array<Scalar, 1, 3> RowArray3;
        typedef Eigen::Translation<Scalar, 3> Translation3;
        // Get cone with base of ellipse and vertex at [0 0 -f]
        // Safaee-Rad 1992 eq (3)
        Conic conic(ellipse);
        Vector3 cam_center_in_ellipse(0, 0, -focal_length);
        Conicoid pupil_cone(conic, cam_center_in_ellipse);
        auto a = pupil_cone.A;
        auto b = pupil_cone.B;
        auto c = pupil_cone.C;
        auto f = pupil_cone.F;
        auto g = pupil_cone.G;
        auto h = pupil_cone.H;
        auto u = pupil_cone.U;
        auto v = pupil_cone.V;
        auto w = pupil_cone.W;
        //auto d = pupil_cone.D;
        // Get canonical conic form:
        //     lambda(1) X^2 + lambda(2) Y^2 + lambda(3) Z^2 = mu
        // Safaee-Rad 1992 eq (6)
        // Done by solving the discriminating cubic (10)
        // Lambdas are sorted descending because order of roots doesn't
        // matter, and it later eliminates the case of eq (30), where
        // lambda(2) > lambda(1)
        RowArray3 lambda;
        std::tie(lambda(0), lambda(1), lambda(2)) = solve(1., -(a + b + c), (b * c + c * a + a * b - f * f - g * g - h * h), -(a * b * c + 2 * f * g * h - a * f * f - b * g * g - c * h * h));
        assert(lambda(0) >= lambda(1));
        assert(lambda(1) > 0);
        assert(lambda(2) < 0);
        // Now want to calculate l,m,n of the plane
        //     lX + mY + nZ = p
        // which intersects the cone to create a circle.
        // Safaee-Rad 1992 eq (31)
        // [Safaee-Rad 1992 eq (33) comes out of this as a result of lambda(1) == lambda(2)]
        auto n = sqrt((lambda(1) - lambda(2)) / (lambda(0) - lambda(2)));
        auto m = 0.0;
        auto l = sqrt((lambda(0) - lambda(1)) / (lambda(0) - lambda(2)));
        // There are two solutions for l, positive and negative, we handle these later
        // Want to calculate T1, the rotation transformation from image
        // space in the canonical conic frame back to image space in the
        // real world
        Matrix3 T1;
        // Safaee-Rad 1992 eq (8)
        auto li = T1.row(0);
        auto mi = T1.row(1);
        auto ni = T1.row(2);
        // Safaee-Rad 1992 eq (12)
        RowArray3 t1 = (b - lambda) * g - f * h;
        RowArray3 t2 = (a - lambda) * f - g * h;
        RowArray3 t3 = -(a - lambda) * (t1 / t2) / g - h / g;
        mi = 1 / sqrt(1 + (t1 / t2).square() + t3.square());
        li = (t1 / t2) * mi.array();
        ni = t3 * mi.array();

        // If li,mi,ni follow the left hand rule, flip their signs
        if ((li.cross(mi)).dot(ni) < 0) {
            li = -li;
            mi = -mi;
            ni = -ni;
        }

        // Calculate T2, a translation transformation from the canonical
        // conic frame to the image space in the canonical conic frame
        // Safaee-Rad 1992 eq (14)
        Translation3 T2;
        T2.translation() = -(u * li + v * mi + w * ni).array() / lambda;
        Circle solutions[2];
        Scalar ls[2] = { l, -l };

        for (int i = 0; i < 2; i++) {
            auto l = ls[i];
            // Circle normal in image space (i.e. gaze vector)
            Vector3 gaze = T1 * Vector3(l, m, n);
            // Calculate T3, a rotation from a frame where Z is the circle normal
            // to the canonical conic frame
            // Safaee-Rad 1992 eq (19)
            // Want T3 = / -m/sqrt(l*l+m*m) -l*n/sqrt(l*l+m*m) l \
            //              |  l/sqrt(l*l+m*m) -m*n/sqrt(l*l+m*m) m |
            //                \            0           sqrt(l*l+m*m)   n /
            // But m = 0, so this simplifies to
            //      T3 = /       0      -n*l/sqrt(l*l) l \
            //              |  l/sqrt(l*l)        0       0 |
            //                \          0         sqrt(l*l)   n /
            //         = /    0    -n*sgn(l) l \
            //              |  sgn(l)     0     0 |
            //                \       0       |l|    n /
            Matrix3 T3;

            if (l == 0) {
                // Discontinuity of sgn(l), have to handle explicitly
                assert(n == 1);
                std::cout << "Warning: l == 0" << std::endl;
                T3 << 0, -1, 0,
                1, 0, 0,
                0, 0, 1;

            } else {
                //auto sgnl = sign(l);
                T3 << 0, -n* sign(l), l,
                sign(l), 0, 0,
                0, abs(l), n;
            }

            // Calculate the circle center
            // Safaee-Rad 1992 eq (38), using T3 as defined in (36)
            auto A = lambda.matrix().dot(T3.col(0).cwiseAbs2());
            auto B = lambda.matrix().dot(T3.col(0).cwiseProduct(T3.col(2)));
            auto C = lambda.matrix().dot(T3.col(1).cwiseProduct(T3.col(2)));
            auto D = lambda.matrix().dot(T3.col(2).cwiseAbs2());
            // Safaee-Rad 1992 eq (41)
            Vector3 center_in_Xprime;
            center_in_Xprime(2) = A * circle_radius / sqrt(sq(B) + sq(C) - A * D);
            center_in_Xprime(0) = -B / A * center_in_Xprime(2);
            center_in_Xprime(1) = -C / A * center_in_Xprime(2);
            // Safaee-Rad 1992 eq (34)
            Translation3 T0;
            T0.translation() << 0, 0, focal_length;
            // Safaee-Rad 1992 eq (42) using (35)
            Vector3 center = T0 * T1 * T2 * T3 * center_in_Xprime;

            // If z is negative (behind the camera), choose the other
            // solution of eq (41) [maybe there's a way of calculating which
            // solution should be chosen first]

            if (center(2) < 0) {
                center_in_Xprime = -center_in_Xprime;
                center = T0 * T1 * T2 * T3 * center_in_Xprime;
            }

            // Make sure that the gaze vector is toward the camera and is normalised
            if (gaze.dot(center) > 0) {
                gaze = -gaze;
            }

            gaze.normalize();
            // Save the results
            solutions[i] = Circle(center, gaze, circle_radius);
        }

        return std::make_pair(solutions[0], solutions[1]);
    }

}

#endif // singleeyefitter_project_h__
