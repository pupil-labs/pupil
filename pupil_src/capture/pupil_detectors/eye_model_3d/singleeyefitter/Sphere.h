#ifndef _SPHERE_H_
#define _SPHERE_H_

#include <Eigen/Core>

namespace singleeyefitter {

    template<typename T>
    class Sphere {
    public:
        typedef T Scalar;
        typedef Eigen::Matrix<Scalar, 3, 1> Vector;

        Vector centre;
        Scalar radius;

        Sphere() : centre(0, 0, 0), radius(0)
        {
        }
        Sphere(Vector centre, Scalar radius)
            : centre(std::move(centre)), radius(std::move(radius))
        {
        }

        static const Sphere Null;

    private:
        // Safe bool stuff
        typedef void (Sphere::*safe_bool_type)() const;
        void this_type_does_not_support_comparisons() const {}
    public:
        operator safe_bool_type() const {
            return *this != Null ? &Sphere::this_type_does_not_support_comparisons : 0;
        }
    };

    template<typename Scalar>
    const Sphere<Scalar> Sphere<Scalar>::Null = Sphere<Scalar>();

    template<typename Scalar>
    bool operator== (const Sphere<Scalar>& s1, const Sphere<Scalar>& s2) {
        return s1.centre == s2.centre
            && s1.radius == s2.radius;
    }
    template<typename Scalar>
    bool operator!= (const Sphere<Scalar>& s1, const Sphere<Scalar>& s2) {
        return s1.centre != s2.centre
            || s1.radius != s2.radius;
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Sphere<T>& circle) {
        return os << "Sphere { centre: (" << circle.centre[0] << "," << circle.centre[1] << "," << circle.centre[2] << "), "
            "radius: " << circle.radius << " }";
    }

}

/*namespace matlab {
    template<typename T>
    struct matlab_traits<typename Sphere<T>, typename std::enable_if<
        matlab::internal::mxHelper<T>::exists
    >::type>
    {
        static Sphere<T> fromMxArray(mxArray* arr) {
            auto arr_size = mxGetNumberOfElements(arr);
            auto arr_ndims = mxGetNumberOfDimensions(arr);
            auto arr_dims = mxGetDimensions(arr);

            if (!mxIsStruct(arr)) {
                throw std::exception("Conversion requires struct");
            }

            mxArray* centre_arr = mxGetField(arr, 0, "centre");
            if (!centre_arr)  {
                throw std::exception("No 'centre' field on struct");
            }

            mxArray* radius_arr = mxGetField(arr, 0, "radius");
            if (!radius_arr)  {
                throw std::exception("No 'radius' field on struct");
            }

            return Sphere<T>(matlab::matlab_traits<Sphere<T>::Vector>::fromMxArray(centre_arr),
                matlab::matlab_traits<Sphere<T>::Scalar>::fromMxArray(radius_arr));
        }

    };
}*/

#endif//_SPHERE_H_
