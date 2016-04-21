#ifndef _ELLIPSE_H_
#define _ELLIPSE_H_

#include "common/constants.h"

#include <Eigen/Core>

namespace singleeyefitter {

    template<typename T>
    class Conic;

    template<typename T>
    class Ellipse2D {
        public:
            typedef T Scalar;
            typedef Eigen::Matrix<Scalar, 2, 1> Vector;
            Vector center;
            Scalar major_radius;
            Scalar minor_radius;
            Scalar angle;

            Ellipse2D()
                : center(0, 0), major_radius(0), minor_radius(0), angle(0)
            {
            }
            template<typename Derived>
            Ellipse2D(const Eigen::EigenBase<Derived>& center, Scalar major_radius, Scalar minor_radius, Scalar angle)
                : center(center), major_radius(major_radius), minor_radius(minor_radius), angle(angle)
            {
            }
            Ellipse2D(Scalar x, Scalar y, Scalar major_radius, Scalar minor_radius, Scalar angle)
                : center(x, y), major_radius(major_radius), minor_radius(minor_radius), angle(angle)
            {
            }
            template<typename U>
            explicit Ellipse2D(const Conic<U>& conic)
            {
                using std::atan2;
                using std::sin;
                using std::cos;
                using std::sqrt;
                using std::abs;
                angle = 0.5 * atan2(conic.B, conic.A - conic.C);
                auto cost = cos(angle);
                auto sint = sin(angle);
                auto sin_squared = sint * sint;
                auto cos_squared = cost * cost;
                auto Ao = conic.F;
                auto Au = conic.D * cost + conic.E * sint;
                auto Av = -conic.D * sint + conic.E * cost;
                auto Auu = conic.A * cos_squared + conic.C * sin_squared + conic.B * sint * cost;
                auto Avv = conic.A * sin_squared + conic.C * cos_squared - conic.B * sint * cost;
                // ROTATED = [Ao Au Av Auu Avv]
                auto tucenter = -Au / (2.0 * Auu);
                auto tvcenter = -Av / (2.0 * Avv);
                auto wcenter = Ao - Auu * tucenter * tucenter - Avv * tvcenter * tvcenter;
                center[0] = tucenter * cost - tvcenter * sint;
                center[1] = tucenter * sint + tvcenter * cost;
                major_radius = sqrt(abs(-wcenter / Auu));
                minor_radius = sqrt(abs(-wcenter / Avv));

                if (major_radius < minor_radius) {
                    std::swap(major_radius, minor_radius);
                    angle = angle + constants::PI / 2;
                }

                if (angle > constants::PI )
                    angle = angle - constants::PI ;
            }

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF((sizeof(Vector) % 16) == 0)

            Vector major_axis() const
            {
                using std::sin;
                using std::cos;
                return Vector(major_radius * sin(angle), major_radius * cos(angle));
            }
            Vector minor_axis() const
            {
                using std::sin;
                using std::cos;
                return Vector(-minor_radius * cos(angle), minor_radius * sin(angle));
            }

            Scalar circumference() const
            {
                using std::abs;
                using std::sqrt;
                using std::pow;
                return constants::PI * abs(3.0 * (major_radius + minor_radius) -
                                  sqrt(10.0 * major_radius * minor_radius + 3.0 *
                                       (pow(major_radius, 2) + pow(minor_radius, 2))));
            }
            Scalar area() const
            {
                return constants::PI * major_radius * minor_radius;
            }


            static const Ellipse2D Null;

        private:

    };

    template<typename Scalar>
    const Ellipse2D<Scalar> Ellipse2D<Scalar>::Null = Ellipse2D<Scalar>();

    template<typename T, typename U>
    bool operator==(const Ellipse2D<T>& el1, const Ellipse2D<U>& el2)
    {
        return el1.center[0] == el2.center[0] &&
               el1.center[1] == el2.center[1] &&
               el1.major_radius == el2.major_radius &&
               el1.minor_radius == el2.minor_radius &&
               el1.angle == el2.angle;
    }
    template<typename T, typename U>
    bool operator!=(const Ellipse2D<T>& el1, const Ellipse2D<U>& el2)
    {
        return !(el1 == el2);
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Ellipse2D<T>& ellipse)
    {
        return os << "Ellipse { center: (" << ellipse.center[0] << "," << ellipse.center[1] << "), a: " <<
               ellipse.major_radius << ", b: " << ellipse.minor_radius << ", theta: " << (ellipse.angle / constants::PI) << "pi }";
    }

    template<typename T, typename U>
    Ellipse2D<T> scaled(const Ellipse2D<T>& ellipse, U scale)
    {
        return Ellipse2D<T>(
                   ellipse.center[0].a,
                   ellipse.center[1].a,
                   ellipse.major_radius.a,
                   ellipse.minor_radius.a,
                   ellipse.angle.a);
    }

    template<class Scalar, class Scalar2>
    inline Eigen::Matrix<typename std::common_type<Scalar, Scalar2>::type, 2, 1> pointAlongEllipse(const Ellipse2D<Scalar>& el, Scalar2 t)
    {
        using std::sin;
        using std::cos;
        auto xt = el.center.x() + el.major_radius * cos(el.angle) * cos(t) - el.minor_radius * sin(el.angle) * sin(t);
        auto yt = el.center.y() + el.major_radius * sin(el.angle) * cos(t) + el.minor_radius * cos(el.angle) * sin(t);
        return Eigen::Matrix<typename std::common_type<Scalar, Scalar2>::type, 2, 1>(xt, yt);
    }

}

#endif
