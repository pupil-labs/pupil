
#ifndef singleeyefitter_ellipsedistanceapproxcalculator_h__
#define singleeyefitter_ellipsedistanceapproxcalculator_h__


#include "common/traits.h"
#include "mathHelper.h"

// Calculates:
//     r * (1 - ||A(p - t)||)
//
//          ||A(p - t)||   maps the ellipse to a unit circle
//      1 - ||A(p - t)||   measures signed distance from unit circle edge
// r * (1 - ||A(p - t)||)  scales this to major radius of ellipse, for (roughly) pixel distance
//
// Actually use (r - ||rAp - rAt||) and precalculate r, rA and rAt.

namespace singleeyefitter {

    using math::norm;

    template<typename T>
    class EllipseDistCalculator {
        public:
            typedef typename ad_traits<T>::scalar Const;

            EllipseDistCalculator(const Ellipse2D<T>& ellipse) : r(ellipse.major_radius)
            {
                using std::sin;
                using std::cos;
                rA << r* cos(ellipse.angle) / ellipse.major_radius, r* sin(ellipse.angle) / ellipse.major_radius,
                -r* sin(ellipse.angle) / ellipse.minor_radius, r* cos(ellipse.angle) / ellipse.minor_radius;
                rAt = rA * ellipse.center;
            }
            template<typename U>
            T operator()(U&& x, U&& y)
            {
                return calculate(std::forward<U>(x), std::forward<U>(y), typename ad_traits<T>::ad_tag(), typename ad_traits<U>::ad_tag());
            }

            template<typename U>
            T calculate(U&& x, U&& y, scalar_tag, scalar_tag)
            {
                T rAxt((rA(0, 0) * x + rA(0, 1) * y) - rAt[0]);
                T rAyt((rA(1, 0) * x + rA(1, 1) * y) - rAt[1]);
                T xy_dist = norm(rAxt, rAyt);
                return (r - xy_dist);
            }

            // Expanded versions for Jet calculations so that Eigen can do some of its expression magic
            template<typename U>
            T calculate(U&& x, U&& y, scalar_tag, ceres_jet_tag)
            {
                T rAxt(rA(0, 0) * x.a + rA(0, 1) * y.a - rAt[0],
                       rA(0, 0) * x.v + rA(0, 1) * y.v);
                T rAyt(rA(1, 0) * x.a + rA(1, 1) * y.a - rAt[1],
                       rA(1, 0) * x.v + rA(1, 1) * y.v);
                T xy_dist = norm(rAxt, rAyt);
                return (r - xy_dist);
            }
            template<typename U>
            T calculate(U&& x, U&& y, ceres_jet_tag, scalar_tag)
            {
                T rAxt(rA(0, 0).a * x + rA(0, 1).a * y - rAt[0].a,
                       rA(0, 0).v * x + rA(0, 1).v * y - rAt[0].v);
                T rAyt(rA(1, 0).a * x + rA(1, 1).a * y - rAt[1].a,
                       rA(1, 0).v * x + rA(1, 1).v * y - rAt[1].v);
                T xy_dist = norm(rAxt, rAyt);
                return (r - xy_dist);
            }
            template<typename U>
            T calculate(U&& x, U&& y, ceres_jet_tag, ceres_jet_tag)
            {
                T rAxt(rA(0, 0).a * x.a + rA(0, 1).a * y.a - rAt[0].a,
                       rA(0, 0).v * x.a + rA(0, 0).a * x.v + rA(0, 1).v * y.a + rA(0, 1).a * y.v - rAt[0].v);
                T rAyt(rA(1, 0).a * x.a + rA(1, 1).a * y.a - rAt[1].a,
                       rA(1, 0).v * x.a + rA(1, 0).a * x.v + rA(1, 1).v * y.a + rA(1, 1).a * y.v - rAt[1].v);
                T xy_dist = norm(rAxt, rAyt);
                return (r - xy_dist);
            }
        private:
            Eigen::Matrix<T, 2, 2> rA;
            Eigen::Matrix<T, 2, 1> rAt;
            T r;
    };

} //namespace

#endif //singleeyefitter_ellipsedistanceapproxcalculator_h__
