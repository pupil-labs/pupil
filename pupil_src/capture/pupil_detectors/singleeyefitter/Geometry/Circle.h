#ifndef _CIRCLE_H_
#define _CIRCLE_H_

#include <Eigen/Core>

namespace singleeyefitter {

    template<typename T>
    class Circle3D {
        public:
            typedef T Scalar;
            typedef Eigen::Matrix<Scalar, 3, 1> Vector;

            Vector center, normal;
            Scalar radius;

            Circle3D() : center(0, 0, 0), normal(0, 0, 0), radius(0)
            {
            }
            Circle3D(Vector center, Vector normal, Scalar radius)
                : center(std::move(center)), normal(std::move(normal)), radius(std::move(radius))
            {
            }

            static const Circle3D Null;

    };

    template<typename Scalar>
    const Circle3D<Scalar> Circle3D<Scalar>::Null = Circle3D<Scalar>();

    template<typename Scalar>
    bool operator== (const Circle3D<Scalar>& s1, const Circle3D<Scalar>& s2)
    {
        return s1.center == s2.center
               && s1.normal == s2.normal
               && s1.radius == s2.radius;
    }
    template<typename Scalar>
    bool operator!= (const Circle3D<Scalar>& s1, const Circle3D<Scalar>& s2)
    {
        return s1.center != s2.center
               || s1.normal != s2.normal
               || s1.radius != s2.radius;
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Circle3D<T>& circle)
    {
        return os << "Circle { center: (" << circle.center[0] << "," << circle.center[1] << "," << circle.center[2] << "), "
               "normal: (" << circle.normal[0] << "," << circle.normal[1] << "," << circle.normal[2] << "), "
               "radius: " << circle.radius << " }";
    }

}

#endif//_CIRCLE_H_
