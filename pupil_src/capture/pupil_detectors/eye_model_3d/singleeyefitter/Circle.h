#ifndef _CIRCLE_H_
#define _CIRCLE_H_

#include <Eigen/Core>

namespace singleeyefitter {

    template<typename T>
    class Circle3D {
    public:
        typedef T Scalar;
        typedef Eigen::Matrix<Scalar, 3, 1> Vector;

        Vector centre, normal;
        Scalar radius;

        Circle3D() : centre(0, 0, 0), normal(0, 0, 0), radius(0)
        {
        }
        Circle3D(Vector centre, Vector normal, Scalar radius)
            : centre(std::move(centre)), normal(std::move(normal)), radius(std::move(radius))
        {
        }

        static const Circle3D Null;

    private:
        // Safe bool stuff
        typedef void (Circle3D::*safe_bool_type)() const;
        void this_type_does_not_support_comparisons() const {}
    public:
        operator safe_bool_type() const {
            return *this != Null ? &Circle3D::this_type_does_not_support_comparisons : 0;
        }
    };

    template<typename Scalar>
    const Circle3D<Scalar> Circle3D<Scalar>::Null = Circle3D<Scalar>();

    template<typename Scalar>
    bool operator== (const Circle3D<Scalar>& s1, const Circle3D<Scalar>& s2) {
        return s1.centre == s2.centre
            && s1.normal == s2.normal
            && s1.radius == s2.radius;
    }
    template<typename Scalar>
    bool operator!= (const Circle3D<Scalar>& s1, const Circle3D<Scalar>& s2) {
        return s1.centre != s2.centre
            || s1.normal != s2.normal
            || s1.radius != s2.radius;
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Circle3D<T>& circle) {
        return os << "Circle { centre: (" << circle.centre[0] << "," << circle.centre[1] << "," << circle.centre[2] << "), "
            "normal: (" << circle.normal[0] << "," << circle.normal[1] << "," << circle.normal[2] << "), "
            "radius: " << circle.radius << " }";
    }

}

/*namespace matlab {
template<typename Scalar>
struct matlab_traits<typename Circle<Scalar>, typename std::enable_if<
    matlab::internal::mxHelper<Scalar>::exists
>::type>
{
    static Circle<Scalar> fromMxArray(mxArray* arr) {
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

        mxArray* normal_arr = mxGetField(arr, 0, "normal");
        if (!normal_arr)  {
            throw std::exception("No 'normal' field on struct");
        }

        mxArray* radius_arr = mxGetField(arr, 0, "radius");
        if (!radius_arr)  {
            throw std::exception("No 'radius' field on struct");
        }

        Circle<Scalar> ret(matlab::matlab_traits<Circle<Scalar>::Vector>::fromMxArray(centre_arr),
                           matlab::matlab_traits<Circle<Scalar>::Vector>::fromMxArray(normal_arr),
                           matlab::matlab_traits<Circle<Scalar>::Scalar>::fromMxArray(radius_arr));
        return ret;
    }

    static std::unique_ptr<mxArray, decltype(&mxDestroyArray)> createMxArray(const Circle<Scalar>& circle) throw() {
        const char* fields[3] = {"centre", "normal", "radius"};
        mwSize dims[1] = {1};
        std::unique_ptr<mxArray, decltype(&mxDestroyArray)> arr(
            mxCreateStructArray(1, dims, 3, fields),
            mxDestroyArray);

        auto centre = matlab::matlab_traits<Circle<Scalar>::Vector>::createMxArray(circle.centre);
        mxSetField(arr.get(), 0, "centre", centre.release());
        auto normal = matlab::matlab_traits<Circle<Scalar>::Vector>::createMxArray(circle.normal);
        mxSetField(arr.get(), 0, "normal", normal.release());
        auto radius = matlab::matlab_traits<Circle<Scalar>::Scalar>::createMxArray(circle.radius);
        mxSetField(arr.get(), 0, "radius", radius.release());

        return arr;
    }
};
}*/

#endif//_CIRCLE_H_
