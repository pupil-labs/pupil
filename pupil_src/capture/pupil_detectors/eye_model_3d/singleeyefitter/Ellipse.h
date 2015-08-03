#ifndef _ELLIPSE_H_
#define _ELLIPSE_H_

#include <boost/math/constants/constants.hpp>
#include <Eigen/Core>

namespace singleeyefitter {

    template<typename T>
    class Conic;

    template<typename T>
    class Ellipse2D {
    public:
        typedef T Scalar;
        typedef Eigen::Matrix<Scalar, 2, 1> Vector;
        Vector centre;
        Scalar major_radius;
        Scalar minor_radius;
        Scalar angle;

        Ellipse2D()
            : centre(0, 0), major_radius(0), minor_radius(0), angle(0)
        {
        }
        template<typename Derived>
        Ellipse2D(const Eigen::EigenBase<Derived>& centre, Scalar major_radius, Scalar minor_radius, Scalar angle)
            : centre(centre), major_radius(major_radius), minor_radius(minor_radius), angle(angle)
        {
        }
        Ellipse2D(Scalar x, Scalar y, Scalar major_radius, Scalar minor_radius, Scalar angle)
            : centre(x, y), major_radius(major_radius), minor_radius(minor_radius), angle(angle)
        {
        }
        template<typename U>
        explicit Ellipse2D(const Conic<U>& conic) {
            using std::atan2;
            using std::sin;
            using std::cos;
            using std::sqrt;
            using std::abs;

            angle = 0.5*atan2(conic.B, conic.A - conic.C);
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

            auto tuCentre = -Au / (2.0*Auu);
            auto tvCentre = -Av / (2.0*Avv);
            auto wCentre = Ao - Auu*tuCentre*tuCentre - Avv*tvCentre*tvCentre;

            centre[0] = tuCentre * cost - tvCentre * sint;
            centre[1] = tuCentre * sint + tvCentre * cost;

            major_radius = sqrt(abs(-wCentre / Auu));
            minor_radius = sqrt(abs(-wCentre / Avv));

            if (major_radius < minor_radius) {
                std::swap(major_radius, minor_radius);
                angle = angle + boost::math::double_constants::pi / 2;
            }
            if (angle > boost::math::double_constants::pi)
                angle = angle - boost::math::double_constants::pi;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF((sizeof(Vector) % 16) == 0)

        Vector major_axis() const {
            using std::sin;
            using std::cos;
            return Vector(major_radius*sin(angle), major_radius*cos(angle));
        }
        Vector minor_axis() const {
            using std::sin;
            using std::cos;
            return Vector(-minor_radius*cos(angle), minor_radius*sin(angle));
        }

        static const Ellipse2D Null;

    private:
        // Safe bool stuff
        typedef void (Ellipse2D::*safe_bool_type)() const;
        void this_type_does_not_support_comparisons() const {}
    public:
        operator safe_bool_type() const {
            return *this != Null ? &Ellipse2D::this_type_does_not_support_comparisons : 0;
        }
    };

    template<typename Scalar>
    const Ellipse2D<Scalar> Ellipse2D<Scalar>::Null = Ellipse2D<Scalar>();

    template<typename T, typename U>
    bool operator==(const Ellipse2D<T>& el1, const Ellipse2D<U>& el2) {
        return el1.centre[0] == el2.centre[0] &&
            el1.centre[1] == el2.centre[1] &&
            el1.major_radius == el2.major_radius &&
            el1.minor_radius == el2.minor_radius &&
            el1.angle == el2.angle;
    }
    template<typename T, typename U>
    bool operator!=(const Ellipse2D<T>& el1, const Ellipse2D<U>& el2) {
        return !(el1 == el2);
    }

    template<typename T>
    std::ostream& operator<< (std::ostream& os, const Ellipse2D<T>& ellipse) {
        return os << "Ellipse { centre: (" << ellipse.centre[0] << "," << ellipse.centre[1] << "), a: " <<
            ellipse.major_radius << ", b: " << ellipse.minor_radius << ", theta: " << (ellipse.angle / boost::math::double_constants::pi) << "pi }";
    }

    template<typename T, typename U>
    Ellipse2D<T> scaled(const Ellipse2D<T>& ellipse, U scale) {
        return Ellipse2D<T>(
            ellipse.centre[0].a,
            ellipse.centre[1].a,
            ellipse.major_radius.a,
            ellipse.minor_radius.a,
            ellipse.angle.a);
    }

    template<class Scalar, class Scalar2>
    inline Eigen::Matrix<typename std::common_type<Scalar, Scalar2>::type, 2, 1> pointAlongEllipse(const Ellipse2D<Scalar>& el, Scalar2 t)
    {
        using std::sin;
        using std::cos;
        auto xt = el.centre.x() + el.major_radius*cos(el.angle)*cos(t) - el.minor_radius*sin(el.angle)*sin(t);
        auto yt = el.centre.y() + el.major_radius*sin(el.angle)*cos(t) + el.minor_radius*cos(el.angle)*sin(t);
        return Eigen::Matrix<typename std::common_type<Scalar, Scalar2>::type, 2, 1>(xt, yt);
    }

}

/*namespace matlab {
template<typename T>
struct matlab_traits<typename Ellipse<T>, typename std::enable_if<
    matlab::internal::mxHelper<T>::exists
>::type>
{
    static std::unique_ptr<mxArray, decltype(&mxDestroyArray)> createMxArray(const Ellipse<T>& ellipse) throw() {
        std::unique_ptr<mxArray, decltype(&mxDestroyArray)> arr(
            internal::mxHelper<T>::createMatrix(1, 5),
            mxDestroyArray);
        T* data = (T*)mxGetData(arr.get());
        data[0] = ellipse.centre[0];
        data[1] = ellipse.centre[1];
        data[2] = ellipse.major_radius;
        data[3] = ellipse.minor_radius;
        data[4] = ellipse.angle;
        return arr;
    }
};
}*/

#endif
