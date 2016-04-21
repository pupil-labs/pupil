#ifndef _SPHERE_H_
#define _SPHERE_H_

#include <Eigen/Core>

namespace singleeyefitter {

    template<typename T>
    class Sphere {
        public:
            typedef T Scalar;
            typedef Eigen::Matrix<Scalar, 3, 1> Vector;

            Vector center;
            Scalar radius;

            Sphere() : center(0, 0, 0), radius(0)
            {
            }
            Sphere(Vector center, Scalar radius)
                : center(std::move(center)), radius(std::move(radius))
            {
            }

            static const Sphere Null;

        private:

    };

    template<typename Scalar>
    const Sphere<Scalar> Sphere<Scalar>::Null = Sphere<Scalar>();

    template<typename Scalar>
    bool operator== (const Sphere<Scalar>& s1, const Sphere<Scalar>& s2)
    {
        return s1.center == s2.center
               && s1.radius == s2.radius;
    }
    template<typename Scalar>
    bool operator!= (const Sphere<Scalar>& s1, const Sphere<Scalar>& s2)
    {
        return s1.center != s2.center
               || s1.radius != s2.radius;
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Sphere<T>& circle)
    {
        return os << "Sphere { center: (" << circle.center[0] << "," << circle.center[1] << "," << circle.center[2] << "), "
               "radius: " << circle.radius << " }";
    }

}

#endif//_SPHERE_H_
