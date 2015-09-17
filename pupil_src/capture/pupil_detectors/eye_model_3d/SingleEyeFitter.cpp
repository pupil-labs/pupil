// SingleEyeFitter.cpp : Defines the entry point for the console application.
//

#include <boost/math/special_functions/sign.hpp>

#include <Eigen/StdVector>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>

#include <singleeyefitter/singleeyefitter.h>

#include <singleeyefitter/utils.h>
#include <singleeyefitter/cvx.h>
#include <singleeyefitter/Conic.h>
#include <singleeyefitter/Ellipse.h>
#include <singleeyefitter/Circle.h>
#include <singleeyefitter/Conicoid.h>
#include <singleeyefitter/Sphere.h>
#include <singleeyefitter/solve.h>
#include <singleeyefitter/intersect.h>
#include <singleeyefitter/projection.h>
#include <singleeyefitter/fun.h>
#include <singleeyefitter/math.h>

#include "distance.h"

#include <spii/spii.h>
#include <spii/term.h>
#include <spii/function.h>
#include <spii/solver.h>

namespace ceres {
    using singleeyefitter::math::sq;

    template<typename T, int N>
    inline Jet<T,N> sq(Jet<T,N> val) {
        val.v *= 2*val.a;
        val.a *= val.a;
        return val;
    }
}

namespace singleeyefitter {

struct scalar_tag{};
struct ceres_jet_tag{};

template<typename T, typename Enabled=void>
struct ad_traits;

template<typename T>
struct ad_traits<T, typename std::enable_if< std::is_arithmetic<T>::value >::type >
{
    typedef scalar_tag ad_tag;
    typedef T scalar;
    static inline scalar value(const T& x) { return x; }
};

template<typename T, int N>
struct ad_traits<::ceres::Jet<T,N>>
{
    typedef ceres_jet_tag ad_tag;
    typedef T scalar;
    static inline scalar get(const ::ceres::Jet<T,N>& x) { return x.a; }
};

template<typename T>
struct ad_traits<T, typename std::enable_if< !std::is_same<T, typename std::decay<T>::type>::value >::type >
    : public ad_traits<typename std::decay<T>::type>
{
};

template<typename T>
inline T smootherstep(T edge0, T edge1, T x, scalar_tag)
{
    if (x >= edge1)
        return T(1);
    else if (x <= edge0)
        return T(0);
    else {
        x = (x - edge0)/(edge1 - edge0);
        return x*x*x*(x*(x*T(6) - T(15)) + T(10));
    }
}
template<typename T, int N>
inline ::ceres::Jet<T,N> smootherstep(T edge0, T edge1, const ::ceres::Jet<T,N>& f, ceres_jet_tag)
{
    if (f.a >= edge1)
        return ::ceres::Jet<T,N>(1);
    else if (f.a <= edge0)
        return ::ceres::Jet<T,N>(0);
    else {
        T x = (f.a - edge0)/(edge1 - edge0);

        // f is referenced by this function, so create new value for return.
        ::ceres::Jet<T,N> g;
        g.a = x*x*x*(x*(x*T(6) - T(15)) + T(10));
        g.v = f.v * (x*x*(x*(x*T(30) - T(60)) + T(30))/(edge1 - edge0));
        return g;
    }
}
template<typename T, int N>
inline ::ceres::Jet<T,N> smootherstep(T edge0, T edge1, ::ceres::Jet<T,N>&& f, ceres_jet_tag)
{
    if (f.a >= edge1)
        return ::ceres::Jet<T,N>(1);
    else if (f.a <= edge0)
        return ::ceres::Jet<T,N>(0);
    else {
        T x = (f.a - edge0)/(edge1 - edge0);

        // f is moved into this function, so reuse it.
        f.a = x*x*x*(x*(x*T(6) - T(15)) + T(10));
        f.v *= (x*x*(x*(x*T(30) - T(60)) + T(30))/(edge1 - edge0));
        return f;
    }
}
template<typename T>
inline auto smootherstep(typename ad_traits<T>::scalar edge0, typename ad_traits<T>::scalar edge1, T&& val)
    -> decltype(smootherstep(edge0, edge1, std::forward<T>(val), typename ad_traits<T>::ad_tag()))
{
    return smootherstep(edge0, edge1, std::forward<T>(val), typename ad_traits<T>::ad_tag());
}

template<typename T>
inline T norm(T x, T y, scalar_tag) {
    using std::sqrt;
    using math::sq;

    return sqrt(sq(x) + sq(y));
}
template<typename T, int N>
inline ::ceres::Jet<T,N> norm(const ::ceres::Jet<T,N>& x, const ::ceres::Jet<T,N>& y, ceres_jet_tag) {
    T anorm = norm<T>(x.a, y.a, scalar_tag());

    ::ceres::Jet<T,N> g;
    g.a = anorm;
    g.v = (x.a/anorm)*x.v + (y.a/anorm)*y.v;

    return g;
}
template<typename T>
inline typename std::decay<T>::type norm(T&& x, T&& y) {
    return norm(std::forward<T>(x), std::forward<T>(y), typename ad_traits<T>::ad_tag());
}

template<typename T>
inline auto Heaviside(T&& val, typename ad_traits<T>::scalar epsilon) -> decltype(smootherstep(-epsilon, epsilon, std::forward<T>(val))) {
    return smootherstep(-epsilon, epsilon, std::forward<T>(val));
}

template<typename Scalar>
cv::Rect bounding_box(const Ellipse2D<Scalar>& ellipse) {
    using std::sin;
    using std::cos;
    using std::sqrt;
    using std::floor;
    using std::ceil;

    Scalar ux = ellipse.major_radius * cos(ellipse.angle);
    Scalar uy = ellipse.major_radius * sin(ellipse.angle);
    Scalar vx = ellipse.minor_radius * cos(ellipse.angle + PI/2);
    Scalar vy = ellipse.minor_radius * sin(ellipse.angle + PI/2);

    Scalar bbox_halfwidth = sqrt(ux*ux + vx*vx);
    Scalar bbox_halfheight = sqrt(uy*uy + vy*vy);

    return cv::Rect(floor(ellipse.centre[0] - bbox_halfwidth), floor(ellipse.centre[1] - bbox_halfheight),
                    2*ceil(bbox_halfwidth) + 1, 2*ceil(bbox_halfheight) + 1);
}

// Calculates:
//     r * (1 - ||A(p - t)||)
//
//          ||A(p - t)||   maps the ellipse to a unit circle
//      1 - ||A(p - t)||   measures signed distance from unit circle edge
// r * (1 - ||A(p - t)||)  scales this to major radius of ellipse, for (roughly) pixel distance
//
// Actually use (r - ||rAp - rAt||) and precalculate r, rA and rAt.
template<typename T>
class EllipseDistCalculator {
public:
    typedef typename ad_traits<T>::scalar Const;

    EllipseDistCalculator(const Ellipse2D<T>& ellipse) : r(ellipse.major_radius)
    {
        using std::sin;
        using std::cos;
        rA << r*cos(ellipse.angle)/ellipse.major_radius, r*sin(ellipse.angle)/ellipse.major_radius,
             -r*sin(ellipse.angle)/ellipse.minor_radius, r*cos(ellipse.angle)/ellipse.minor_radius;
        rAt = rA*ellipse.centre;
    }
    template<typename U>
    T operator()(U&& x, U&& y) {
        return calculate(std::forward<U>(x), std::forward<U>(y), typename ad_traits<T>::ad_tag(), typename ad_traits<U>::ad_tag());
    }

    template<typename U>
    T calculate(U&& x, U&& y, scalar_tag, scalar_tag) {
        T rAxt((rA(0,0) * x + rA(0,1) * y) - rAt[0]);
        T rAyt((rA(1,0) * x + rA(1,1) * y) - rAt[1]);

        T xy_dist = norm(rAxt, rAyt);

        return (r - xy_dist);
    }

    // Expanded versions for Jet calculations so that Eigen can do some of its expression magic
    template<typename U>
    T calculate(U&& x, U&& y, scalar_tag, ceres_jet_tag) {
        T rAxt(rA(0,0) * x.a + rA(0,1) * y.a - rAt[0],
            rA(0,0) * x.v + rA(0,1) * y.v);
        T rAyt(rA(1,0) * x.a + rA(1,1) * y.a - rAt[1],
            rA(1,0) * x.v + rA(1,1) * y.v);

        T xy_dist = norm(rAxt, rAyt);

        return (r - xy_dist);
    }
    template<typename U>
    T calculate(U&& x, U&& y, ceres_jet_tag, scalar_tag) {
        T rAxt(rA(0,0).a * x + rA(0,1).a * y - rAt[0].a,
            rA(0,0).v * x + rA(0,1).v * y - rAt[0].v);
        T rAyt(rA(1,0).a * x + rA(1,1).a * y - rAt[1].a,
            rA(1,0).v * x + rA(1,1).v * y - rAt[1].v);

        T xy_dist = norm(rAxt, rAyt);

        return (r - xy_dist);
    }
    template<typename U>
    T calculate(U&& x, U&& y, ceres_jet_tag, ceres_jet_tag) {
        T rAxt(rA(0,0).a * x.a + rA(0,1).a * y.a - rAt[0].a,
            rA(0,0).v * x.a + rA(0,0).a * x.v + rA(0,1).v * y.a + rA(0,1).a * y.v - rAt[0].v);
        T rAyt(rA(1,0).a * x.a + rA(1,1).a * y.a - rAt[1].a,
            rA(1,0).v * x.a + rA(1,0).a * x.v + rA(1,1).v * y.a + rA(1,1).a * y.v - rAt[1].v);

        T xy_dist = norm(rAxt, rAyt);

        return (r - xy_dist);
    }
private:
    Eigen::Matrix<T, 2, 2> rA;
    Eigen::Matrix<T, 2, 1> rAt;
    T r;
};

// Calculates the x crossings of a conic at a given y value. Returns the number of crossings (0, 1 or 2)
template<typename Scalar>
int getXCrossing(const Conic<Scalar>& conic, Scalar y, Scalar& x1, Scalar& x2) {
    using std::sqrt;

    Scalar a = conic.A;
    Scalar b = conic.B*y + conic.D;
    Scalar c = conic.C*y*y + conic.E*y + conic.F;

    Scalar det = b*b - 4*a*c;
    if (det == 0) {
        x1 = -b/(2*a);
        return 1;
    } else if (det < 0) {
        return 0;
    } else {
        Scalar sqrtdet = sqrt(det);
        x1 = (-b - sqrtdet)/(2*a);
        x2 = (-b + sqrtdet)/(2*a);
        return 2;
    }
}

template<template<class, int> class Jet, class T, int N>
typename std::enable_if<std::is_same<typename ad_traits<Jet<T,N>>::ad_tag, ceres_jet_tag>::value, Ellipse2D<T>>::type
toConst(const Ellipse2D<Jet<T,N>>& ellipse) {
    return Ellipse2D<T>(
        ellipse.centre[0].a,
        ellipse.centre[1].a,
        ellipse.major_radius.a,
        ellipse.minor_radius.a,
        ellipse.angle.a);
}

template<class T>
Ellipse2D<T> scaledMajorRadius(const Ellipse2D<T>& ellipse, const T& target_radius) {
    return Ellipse2D<T>(
        ellipse.centre[0],
        ellipse.centre[1],
        target_radius,
        target_radius * ellipse.minor_radius/ellipse.major_radius,
        ellipse.angle);
};

namespace internal {
    template<class T> T ellipseGoodness(const Ellipse2D<T>& ellipse, const cv::Mat_<uint8_t>& eye, T band_width, T step_epsilon, scalar_tag);
    template<class T> T ellipseGoodness(const Ellipse2D<T>& ellipse, const cv::Mat_<uint8_t>& eye, typename ad_traits<T>::scalar band_width, typename ad_traits<T>::scalar step_epsilon, ceres_jet_tag);
}

// Calculates the "goodness" of an ellipse.
//
// This is defined as the difference in region means:
//
//    μ⁻ - μ⁺
//
// where
//         Σ_p (H(d(p)+w) - H(d(p))) I(p)
//    μ⁻ = ------------------------------
//           Σ_p (H(d(p)+w) - H(d(p)))
//
//         Σ_p (H(d(p)+w) - H(d(p))) I(p)
//    μ⁺ = ------------------------------
//           Σ_p (H(d(p)+w) - H(d(p)))
//
// (see eqs 16, 20, 21 in the PETMEI paper)
//
// The ellipse distance d(p) is defined as
//
//    d(p) = r * (1 - ||A(p - t)||)
//
// with r as the major radius and A as the matrix that transforms the ellipse to a unit circle.
//
//          ||A(p - t)||   maps the ellipse to a unit circle
//      1 - ||A(p - t)||   measures signed distance from unit circle edge
// r * (1 - ||A(p - t)||)  scales this to major radius of ellipse, for (roughly) pixel distance
//
template<class T>
inline T ellipseGoodness(const Ellipse2D<T>& ellipse, const cv::Mat_<uint8_t>& eye, typename ad_traits<T>::scalar band_width, typename ad_traits<T>::scalar step_epsilon) {
    // band_width     The width of each band (inner and outer)
    // step_epsilon   The epsilon of the soft step function

    return internal::ellipseGoodness<T>(ellipse, eye, band_width, step_epsilon, typename ad_traits<T>::ad_tag());
}

//#define DEBUG_ELLIPSE_GOODNESS
//#define USE_INLINED_ELLIPSE_DIST

#ifdef USE_INLINED_ELLIPSE_DIST
#define IF_INLINED_ELLIPSE_DIST(...) __VA_ARGS__
#else
#define IF_INLINED_ELLIPSE_DIST(...)
#endif

namespace internal {
// Non autodiff version of ellipse goodness calculation
template<class T>
T ellipseGoodness(const Ellipse2D<T>& ellipse, const cv::Mat_<uint8_t>& eye, T band_width, T step_epsilon, scalar_tag) {
    using std::max;
    using std::min;
    using std::ceil;
    using std::floor;
    using std::sin;
    using std::cos;

    // Ellipses (and corresponding conics) delimiting the region in which the band masks will be non-zero
    Ellipse2D<T> outerEllipse = scaledMajorRadius(ellipse, ellipse.major_radius + ((band_width + step_epsilon) + 0.5));
    Ellipse2D<T> innerEllipse = scaledMajorRadius(ellipse, ellipse.major_radius - ((band_width + step_epsilon) + 0.5));
    Conic<T> outerConic(outerEllipse);
    Conic<T> innerConic(innerEllipse);

    // Variables for calculating the mean
    T sum_inner = T(0), count_inner = T(0), sum_outer = T(0), count_outer = T(0);

    // Only iterate over pixels within the outer ellipse's bounding box
    cv::Rect bb = bounding_box(outerEllipse);
    bb &= cv::Rect(-eye.cols/2,-eye.rows/2,eye.cols,eye.rows);


#ifndef USE_INLINED_ELLIPSE_DIST
    // Ellipse distance calculator
    EllipseDistCalculator<T> ellipDist(ellipse);
#else
    // Instead of calculating
    //     r * (1 - ||A(p - t)||)
    // we use
    //     (r - ||rAp - rAt||)
    // and precalculate r, rA and rAt.
    Eigen::Matrix<T, 2, 2> rA;
    T r = ellipse.major_radius;
    rA << r*cos(ellipse.angle)/ellipse.major_radius, r*sin(ellipse.angle)/ellipse.major_radius,
        -r*sin(ellipse.angle)/ellipse.minor_radius, r*cos(ellipse.angle)/ellipse.minor_radius;
    Eigen::Matrix<T, 2, 1> rAt = rA*ellipse.centre;

    // Actually,
    ///    rAp - rAt = rA(0,y) + rA(x,0) - rAt
    // So, can perform a strength reduction to calculate rAp iteratively.

    // rA(0,y) - rAt, with y_0 = bb.y
    Eigen::Matrix<T, 2, 1> rA0yrAt(rA(0,1) * bb.y - rAt[0], rA(1,1) * bb.y - rAt[1]);
    // rA(1,0), for incrementing x
    Eigen::Matrix<T, 2, 1> rA10 = rA.col(0);
    // rA(0,1), for incrementing y
    Eigen::Matrix<T, 2, 1> rA01 = rA.col(1);
#endif

    for (int i = bb.y; i < bb.y + bb.height; ++i IF_INLINED_ELLIPSE_DIST(, rA0yrAt += rA01)) {
        // Image row pointer -- (0,0) is centre of image, so shift accordingly
        const uint8_t* eye_i = eye[i + eye.rows/2];

        // Only iterate over pixels between the inner and outer ellipse
        T ox1, ox2;
        int outerCrossings = getXCrossing<T>(outerConic, i, ox1, ox2);
        if (outerCrossings < 2) {
            // If we don't cross the outer ellipse at all, exit early
            continue;
        }
        T ix1, ix2;
        int innerCrossings = innerEllipse.minor_radius > 0 ? getXCrossing<T>(innerConic, i, ix1, ix2) : 0;

        // Define pairs of x values to iterate between
        std::vector<std::pair<int,int>> xpairs;
        if (innerCrossings < 2) {
            // If we don't cross the inner ellipse, iterate between the two crossings of the outer ellipse
            xpairs.emplace_back(max<int>(floor(ox1),bb.x), min<int>(ceil(ox2), bb.x+bb.width-1));
        } else {
            // Otherwise, iterate between outer-->inner, then inner-->outer.
            xpairs.emplace_back(max<int>(floor(ox1),bb.x), min<int>(ceil(ix1), bb.x+bb.width-1));
            xpairs.emplace_back(max<int>(floor(ix2),bb.x), min<int>(ceil(ox2), bb.x+bb.width-1));
        }

        // Go over x pairs (that is, outer-->outer or outer-->inner,inner-->outer)
        for (const auto& xpair : xpairs) {
            // Pixel pointer, shifted accordingly
            const uint8_t* eye_ij = eye_i + xpair.first + eye.cols/2;

#ifdef USE_INLINED_ELLIPSE_DIST
            // rA(0,y) + rA(x,0) - rAt, with x_0 = xpair.first
            Eigen::Matrix<T, 2, 1> rApt(rA0yrAt(0) + rA(0,0)*xpair.first, rA0yrAt(1) + rA(1,0)*xpair.first);
#endif

            for (int j = xpair.first; j <= xpair.second; ++j, ++eye_ij IF_INLINED_ELLIPSE_DIST(, rApt += rA10)) {
                auto eye_ij_val = *eye_ij;
                if (eye_ij_val > 200) {
                    // Ignore bright areas (i.e. glints)
                    continue;
                }

#ifdef USE_INLINED_ELLIPSE_DIST
                T dist = (r - norm(rApt(0), rApt(1)));
#else
                T dist = ellipDist(T(j), T(i));
#endif

                // Calculate mask values for each band
                T Hellip = Heaviside(dist, step_epsilon);
                T Houter = Heaviside(dist+band_width, step_epsilon);
                T Hinner = Heaviside(dist-band_width, step_epsilon);

                T outer_weight = (Houter - Hellip);
                T inner_weight = (Hellip - Hinner);

                sum_outer += outer_weight * eye_ij_val;
                count_outer += outer_weight;

                sum_inner += inner_weight * eye_ij_val;
                count_inner += inner_weight;
            }
        }
    }

    // Get mean values, defaulting to 255 and 0 if count_inner/count_outer are 0 (respectively)
    // Using 255 and 0 because these are the "worst" values, so some pixels will be preferred over none.
    T mu_inner = (count_inner==0 ? 255 : sum_inner/count_inner);
    T mu_outer = (count_outer==0 ? 0 : sum_outer/count_outer);

    // If count < 100 pixels, interpolate between mean value and "worst" value. This will push the
    // gradient away from small pixel counts in a vaguely smooth way.
    if (count_outer < 100) {
        mu_outer = math::lerp<T>(0, mu_outer, count_outer/100.0);
    }
    if (count_inner < 100) {
        mu_inner = math::lerp<T>(255, mu_inner, count_inner/100.0);
    }

    // Return difference of mean values
    return mu_outer - mu_inner;
}

// Autodiff version of ellipse goodness calculation
template<class Jet>
Jet ellipseGoodness(const Ellipse2D<Jet>& ellipse, const cv::Mat_<uint8_t>& eye, typename ad_traits<Jet>::scalar band_width, typename ad_traits<Jet>::scalar step_epsilon, ceres_jet_tag) {
    using std::max;
    using std::min;
    using std::ceil;
    using std::floor;

#ifdef DEBUG_ELLIPSE_GOODNESS
    cv::Mat_<cv::Vec3b> eye_proc = cv::Mat_<cv::Vec3b>::zeros(eye.rows, eye.cols);
    cv::Mat_<cv::Vec3b> eye_H = cv::Mat_<cv::Vec3b>::zeros(eye.rows, eye.cols);
#endif

    typedef typename ad_traits<Jet>::scalar T;
    typedef Jet Jet_t;

    // A constant version of the ellipse
    Ellipse2D<T> constEllipse = toConst(ellipse);

    // Ellipses (and corresponding conics) delimiting the region in which the band masks will be non-zero
    Ellipse2D<T> constOuterEllipse = scaledMajorRadius(constEllipse, constEllipse.major_radius + ((band_width + step_epsilon) + 0.5));
    Ellipse2D<T> constInnerEllipse = scaledMajorRadius(constEllipse, constEllipse.major_radius - ((band_width + step_epsilon) + 0.5));
    Conic<T> constOuterConic(constOuterEllipse);
    Conic<T> constInnerConic(constInnerEllipse);

    // Variables for calculating the mean
    Jet_t sum_inner = Jet_t(0), count_inner = Jet_t(0), sum_outer = Jet_t(0), count_outer = Jet_t(0);

    // Only iterate over pixels within the outer ellipse's bounding box
    cv::Rect bb = bounding_box(constOuterEllipse);
    bb &= cv::Rect(-eye.cols/2,-eye.rows/2,eye.cols,eye.rows);


#ifndef USE_INLINED_ELLIPSE_DIST
    // Ellipse distance calculator
    EllipseDistCalculator<Jet_t> ellipDist(ellipse);
    EllipseDistCalculator<T> constEllipDist(constEllipse);
#else
    // Instead of calculating
    //     r * (1 - ||A(p - t)||)
    // we use
    //     (r - ||rAp - rAt||)
    // and precalculate r, rA and rAt.
    Eigen::Matrix<T, 2, 2> rA;
    T r = constEllipse.major_radius;
    rA << r*cos(constEllipse.angle)/constEllipse.major_radius, r*sin(constEllipse.angle)/constEllipse.major_radius,
         -r*sin(constEllipse.angle)/constEllipse.minor_radius, r*cos(constEllipse.angle)/constEllipse.minor_radius;
    Eigen::Matrix<T, 2, 1> rAt = rA*constEllipse.centre;

    // And non-constant versions of the above
    Eigen::Matrix<Jet_t, 2, 2> rA_jet;
    Jet_t r_jet = ellipse.major_radius;
    rA_jet << r_jet*cos(ellipse.angle)/ellipse.major_radius, r_jet*sin(ellipse.angle)/ellipse.major_radius,
             -r_jet*sin(ellipse.angle)/ellipse.minor_radius, r_jet*cos(ellipse.angle)/ellipse.minor_radius;
    Eigen::Matrix<Jet_t, 2, 1> rAt_jet = rA_jet*ellipse.centre;

    // Actually,
    ///    rAp - rAt = rA(0,y) + rA(x,0) - rAt
    // So, can perform a strength reduction to calculate rAp iteratively.

    // rA(0,y) - rAt, with y_0 = bb.y
    Eigen::Matrix<T, 2, 1> rA0yrAt(rA(0,1) * bb.y - rAt[0], rA(1,1) * bb.y - rAt[1]);
    // rA(1,0), for incrementing x
    Eigen::Matrix<T, 2, 1> rA10 = rA.col(0);
    // rA(0,1), for incrementing y
    Eigen::Matrix<T, 2, 1> rA01 = rA.col(1);
#endif

    for (int i = bb.y; i < bb.y + bb.height; ++i IF_INLINED_ELLIPSE_DIST(, rA0yrAt += rA01)) {
        // Image row pointer -- (0,0) is centre of image, so shift accordingly
        const uint8_t* eye_i = eye[i + eye.rows/2];

        // Only iterate over pixels between the inner and outer ellipse
        T ox1, ox2;
        int outerCrossings = getXCrossing<T>(constOuterConic, i, ox1, ox2);
        if (outerCrossings < 2) {
            // If we don't cross the outer ellipse at all, exit early
            continue;
        }
        T ix1, ix2;
        int innerCrossings = constInnerEllipse.major_radius > 0 ? getXCrossing<T>(constInnerConic, i, ix1, ix2) : 0;

        // Define pairs of x values to iterate between
        std::vector<std::pair<int,int>> xpairs;
        if (innerCrossings < 2) {
            // If we don't cross the inner ellipse, iterate between the two crossings of the outer ellipse
            xpairs.emplace_back(max<int>(floor(ox1),bb.x), min<int>(ceil(ox2), bb.x+bb.width-1));
        } else {
            // Otherwise, iterate between outer-->inner, then inner-->outer.
            xpairs.emplace_back(max<int>(floor(ox1),bb.x), min<int>(ceil(ix1), bb.x+bb.width-1));
            xpairs.emplace_back(max<int>(floor(ix2),bb.x), min<int>(ceil(ox2), bb.x+bb.width-1));
        }

#ifdef USE_INLINED_ELLIPSE_DIST
        // Precalculate the gradient of
        //     rA(y,0) - rAt
        auto rAy0rAt_x_v = (rA_jet(0,1).v * i - rAt_jet(0).v).eval();
        auto rAy0rAt_y_v = (rA_jet(1,1).v * i - rAt_jet(1).v).eval();
#endif

        // Go over x pairs (that is, outer-->outer or outer-->inner,inner-->outer)
        for (const auto& xpair : xpairs) {

            // Pixel pointer, shifted accordingly
            const uint8_t* eye_ij = eye_i + xpair.first + eye.cols/2;

#ifdef USE_INLINED_ELLIPSE_DIST
            // rA(0,y) + rA(x,0) - rAt, with x_0 = xpair.first
            Eigen::Matrix<T, 2, 1> rApt(rA0yrAt(0) + rA(0,0)*xpair.first, rA0yrAt(1) + rA(1,0)*xpair.first);
#endif

            for (int j = xpair.first; j <= xpair.second; ++j, ++eye_ij IF_INLINED_ELLIPSE_DIST(, rApt += rA10)) {

                T eye_ij_val = *eye_ij;
                if (eye_ij_val > 200) {
                    // Ignore bright areas (i.e. glints)
                    continue;
                }

                // Calculate signed ellipse distance without gradient first, in case the gradient is 0
#ifdef USE_INLINED_ELLIPSE_DIST
                T dist_const = (r - norm(rApt(0), rApt(1)));
#else
                T dist_const = constEllipDist(T(j), T(i));
#endif

                // Check if we are within step_epsilon of the edges of the bands. If yes, calculate
                // the gradient. Otherwise, the gradient is known to be 0.
                if (abs(dist_const) < step_epsilon
                    || abs(dist_const-band_width) < step_epsilon
                    || abs(dist_const+band_width) < step_epsilon) {

#ifdef USE_INLINED_ELLIPSE_DIST
                    // Calculate the gradients of rApt, and use those to get the dist
                    Jet_t rAxt_jet(rApt(0),
                        rA_jet(0,0).v * j + rAy0rAt_x_v);
                    Jet_t rAyt_jet(rApt(1),
                        rA_jet(1,0).v * j + rAy0rAt_y_v);

                    //Eigen::Matrix<Jet,2,1> rApt_jet2 = rA_jet*Eigen::Matrix<Jet,2,1>(Jet(j),Jet(i)) - rAt_jet;

                    Jet_t dist = (r_jet - norm(rAxt_jet, rAyt_jet));
                    //Jet_t dist2 = ellipDist(T(j), T(i));
#else
                    Jet_t dist = ellipDist(T(j), T(i));
#endif

                    // Calculate mask values and derivatives for each band
                    Jet_t Hellip = Heaviside(dist, step_epsilon);
                    Jet_t Houter = Heaviside(dist+band_width, step_epsilon);
                    Jet_t Hinner = Heaviside(dist-band_width, step_epsilon);

                    Jet_t outer_weight = (Houter - Hellip);
                    Jet_t inner_weight = (Hellip - Hinner);

                    // Inline the Jet operator+= to allow eigen expression and noalias magic.
                    sum_outer.a += outer_weight.a * eye_ij_val;
                    sum_outer.v.noalias() += outer_weight.v * eye_ij_val;
                    count_outer.a += outer_weight.a;
                    count_outer.v.noalias() += outer_weight.v;

                    sum_inner.a += inner_weight.a * eye_ij_val;
                    sum_inner.v.noalias() += inner_weight.v * eye_ij_val;
                    count_inner.a += inner_weight.a;
                    count_inner.v.noalias() += inner_weight.v;

                    #ifdef DEBUG_ELLIPSE_GOODNESS
                        eye_H(i + eye.rows/2,j + eye.cols/2)[2] = outer_weight.a*255;
                        eye_H(i + eye.rows/2,j + eye.cols/2)[1] = inner_weight.a*255;
                        eye_H(i + eye.rows/2,j + eye.cols/2)[0] = 255;

                        eye_proc(i + eye.rows/2,j + eye.cols/2)[2] = outer_weight.a * eye_ij_val;
                        eye_proc(i + eye.rows/2,j + eye.cols/2)[1] = inner_weight.a * eye_ij_val;
                        eye_proc(i + eye.rows/2,j + eye.cols/2)[0] = 255;
                    #endif

                } else {
                    // Calculate mask values for each band
                    T Hellip = Heaviside(dist_const, step_epsilon);
                    T Houter = Heaviside(dist_const+band_width, step_epsilon);
                    T Hinner = Heaviside(dist_const-band_width, step_epsilon);

                    T outer_weight = (Houter - Hellip);
                    T inner_weight = (Hellip - Hinner);

                    sum_outer.a += outer_weight * eye_ij_val;
                    count_outer.a += outer_weight;

                    sum_inner.a += inner_weight * eye_ij_val;
                    count_inner.a += inner_weight;

                    #ifdef DEBUG_ELLIPSE_GOODNESS
                    eye_H(i + eye.rows/2,j + eye.cols/2)[2] = outer_weight*255;
                    eye_H(i + eye.rows/2,j + eye.cols/2)[1] = inner_weight*255;
                    eye_H(i + eye.rows/2,j + eye.cols/2)[0] = 0;

                    eye_proc(i + eye.rows/2,j + eye.cols/2)[2] = outer_weight * eye_ij_val;
                    eye_proc(i + eye.rows/2,j + eye.cols/2)[1] = inner_weight * eye_ij_val;
                    eye_proc(i + eye.rows/2,j + eye.cols/2)[0] = 255;
                    #endif
                }
            }
        }
    }

    // Get mean values, defaulting to 255 and 0 if count_inner/count_outer are 0 (respectively)
    // Using 255 and 0 because these are the "worst" values, so some pixels will be preferred over none.
    Jet mu_inner = (count_inner.a==0 ? Jet(255) : sum_inner/count_inner);
    Jet mu_outer = (count_outer.a==0 ? Jet(0) : sum_outer/count_outer);

    // If count < 100 pixels, interpolate between mean value and "worst" value. This will push the
    // gradient away from small pixel counts in a vaguely smooth way.
    if (count_outer.a < 100) {
        mu_outer = math::lerp<Jet>(Jet(0), mu_outer, count_outer/100.0);
    }
    if (count_inner.a < 100) {
        mu_inner = math::lerp<Jet>(Jet(255), mu_inner, count_inner/100.0);
    }

    // Return difference of mean values
    return mu_outer - mu_inner;
}
}

template<typename T>
Eigen::Matrix<T,3,1> sph2cart(T r, T theta, T psi) {
    using std::sin;
    using std::cos;

    return r * Eigen::Matrix<T,3,1>(sin(theta)*cos(psi), cos(theta), sin(theta)*sin(psi));
}

template<typename T>
T angleDiffGoodness(T theta1, T psi1, T theta2, T psi2, typename ad_traits<T>::scalar sigma) {
    using std::sin;
    using std::cos;
    using std::acos;
    using std::asin;
    using std::atan2;
    using std::sqrt;

    if (theta1 == theta2 && psi1 == psi2) {
        return T(1);
    }

    // Haversine distance
    auto dist = T(2)*asin(sqrt(sq(sin((theta1-theta2)/T(2))) + cos(theta1)*cos(theta2)*sq(sin((psi1-psi2)/T(2)))));
    return exp(-sq(dist)/sq(sigma));
}

template<typename T>
Circle3D<T> circleOnSphere(const Sphere<T>& sphere, T theta, T psi, T circle_radius) {
    typedef Eigen::Matrix<T,3,1> Vector3;

    Vector3 radial = sph2cart<T>(T(1), theta, psi);
    return Circle3D<T>(sphere.centre + sphere.radius * radial,
        radial,
        circle_radius);
}

template<typename T>
struct EllipseGoodnessFunction {
    T operator()(const Sphere<T>& eye, T theta, T psi, T pupil_radius, T focal_length, typename ad_traits<T>::scalar band_width, typename ad_traits<T>::scalar step_epsilon, const cv::Mat& mEye) {
        typedef Eigen::Matrix<T,3,1> Vector3;
        typedef typename ad_traits<T>::scalar Const;

        static const Vector3 camera_centre(T(0),T(0),T(0));

        // Check for bounds. The worst possible value of ellipseGoodness is -255, so use that as a starting point for out-of-bounds pupils

        // Pupil radius must be positive
        if (pupil_radius <= Const(0))
        {
            // Return -255 for radius == 0, and even lower values for
            // radius < 0
            // This should push the gradient towards positive radius,
            // rather than just returning flat -255
            return Const(-255.0) + pupil_radius;
        }

        Circle3D<T> pupil_circle = circleOnSphere(eye, theta, psi, pupil_radius);

        // Ellipse normal must point towards camera
        T normalDotPos = pupil_circle.normal.dot(camera_centre - pupil_circle.centre);
        if (normalDotPos <= Const(0))
        {
            // Return -255 for normalDotPos == 0, and even lower values for
            // normalDotPos < 0
            // This should push the gradient towards positive normalDotPos,
            // rather than just returning flat -255
            return Const(-255.0) + normalDotPos;
        }

        // Angles should be in the range
        //    theta: 0 -> pi
        //      psi: -pi -> 0
        // If we're outside of this range AND radialDotEye > 0, then we must
        // have gone all the way around, so just return worst case (i.e as bad
        // as radialDotEye == -1) with additional penalty for how far out we
        // are, again to push the gradient back inwards.
        if (theta < Const(0) || theta > Const(PI) || psi < Const(-PI) || psi > Const(0))
        {
            T ret = Const(-255.0) - (camera_centre - pupil_circle.centre).norm();
            if (theta < Const(0))
                ret -= (Const(0) - theta);
            else if (theta > Const(PI))
                ret -= (theta - Const(PI));
            if (psi < Const(-PI))
                ret -= (Const(-PI) - psi);
            else if (psi > Const(0))
                ret -= (psi - Const(0));
        }

        // Ok, everything looks good so far, calculate the actual goodness.

        Ellipse2D<T> pupil_ellipse(project(pupil_circle, focal_length));

        return ellipseGoodness<T>(pupil_ellipse, mEye, band_width, step_epsilon);
    }
};





template<typename Scalar>
class EllipseDistanceResidualFunction {
public:
    EllipseDistanceResidualFunction(const cv::Mat& eye_image, const std::vector<cv::Point2f>& pupil_inliers, const Scalar& eye_radius, const Scalar& focal_length) :
        eye_image(eye_image), pupil_inliers(pupil_inliers), eye_radius(eye_radius), focal_length(focal_length) {}

    template <typename T>
    bool operator()(const T* const eye_param, const T* const pupil_param, T* e) const {
        typedef typename ad_traits<T>::scalar Const;

        Eigen::Matrix<T,3,1> eye_pos(eye_param[0], eye_param[1], eye_param[2]);
        Sphere<T> eye(eye_pos, T(eye_radius));

        Ellipse2D<T> pupil_ellipse(project(circleOnSphere(eye, pupil_param[0], pupil_param[1], pupil_param[2]), T(focal_length)));

        EllipseDistCalculator<T> ellipDist(pupil_ellipse);

        for (int i = 0; i < pupil_inliers.size(); ++i) {
            const cv::Point2f& inlier = pupil_inliers[i];
            e[i] = ellipDist(Const(inlier.x), Const(inlier.y));
        }

        return true;
    }
private:
    const cv::Mat& eye_image;
    const std::vector<cv::Point2f>& pupil_inliers;
    const Scalar& eye_radius;
    const Scalar& focal_length;
};


template<typename Scalar>
struct EllipsePointDistanceFunction {
    EllipsePointDistanceFunction(const Ellipse2D<Scalar>& el, Scalar x, Scalar y) : el(el), x(x), y(y) {}

    template <typename T>
    bool operator()(const T* const t, T* e) const
    {
        using std::sin;
        using std::cos;

        auto&& pt = pointAlongEllipse(el, t[0]);
        e[0] = norm(x - pt.x(), y - pt.y());

        return true;
    }

    const Ellipse2D<Scalar>& el;
    Scalar x, y;
};

template<bool has_eye_var=true>
struct PupilContrastTerm : public spii::Term {
    const Sphere<double>& init_eye;
    double focal_length;
    const cv::Mat eye_image;
    double band_width;
    double step_epsilon;

    int eye_var_idx() const { return has_eye_var ? 0 : -1; }
    int pupil_var_idx() const { return has_eye_var ? 1 : 0; }

    PupilContrastTerm(const Sphere<double>& eye, double focal_length, cv::Mat eye_image, double band_width, double step_epsilon) :
        init_eye(eye),
        focal_length(focal_length),
        eye_image(eye_image),
        band_width(band_width),
        step_epsilon(step_epsilon)
    {}

    virtual int number_of_variables() const override {
        int nvars = 1; // This pupil params
        if (has_eye_var)
            nvars++; // Eye params

        return nvars;
    }
    virtual int variable_dimension(int var) const override {
        if (var == eye_var_idx()) // Eye params (x,y,z)
            return 3;
        if (var == pupil_var_idx()) // This pupil params (theta, psi, r)
            return 3;
        return -1;
    };
    virtual double evaluate(double * const * const vars) const override
    {
        auto& pupil_vars = vars[pupil_var_idx()];

        auto eye = init_eye;
        if (has_eye_var) {
            auto& eye_vars = vars[eye_var_idx()];
            eye.centre = Sphere<double>::Vector(eye_vars[0], eye_vars[1], eye_vars[2]);
        }

        EllipseGoodnessFunction<double> goodnessFunction;
        auto theta = pupil_vars[0];
        auto psi = pupil_vars[1];
        auto r = pupil_vars[2];
        auto goodness = goodnessFunction(eye,
            theta, psi, r,
            focal_length,
            band_width, step_epsilon,
            eye_image);

        return -goodness;
    }
    virtual double evaluate(double * const * const vars, std::vector<Eigen::VectorXd>* gradient) const override
    {
        auto& pupil_vars = vars[pupil_var_idx()];

        double contrast_goodness_a;
        Eigen::Matrix<double,3,1> eye_contrast_goodness_v;
        Eigen::Matrix<double,3,1> pupil_contrast_goodness_v;

        // Get region contrast goodness using EllipseGoodnessFunction.
        if (has_eye_var) {
            // If varying the eye parameters, calculate the gradient wrt. to 6 params (3 eye + 3 pupil)
            typedef ceres::Jet<double, 6> EyePupilJet;

            auto& eye_vars = vars[eye_var_idx()];
            Eigen::Matrix<EyePupilJet,3,1> eye_pos(EyePupilJet(eye_vars[0], 0), EyePupilJet(eye_vars[1], 1), EyePupilJet(eye_vars[2], 2));
            Sphere<EyePupilJet> eye(eye_pos, EyePupilJet(init_eye.radius));

            EyePupilJet contrast_goodness;
            {
                EllipseGoodnessFunction<EyePupilJet> goodnessFunction;
                auto theta = EyePupilJet(pupil_vars[0], 3);
                auto psi = EyePupilJet(pupil_vars[1], 4);
                auto r = EyePupilJet(pupil_vars[2], 5);
                contrast_goodness = goodnessFunction(eye,
                    theta, psi, r,
                    EyePupilJet(focal_length),
                    band_width, step_epsilon,
                    eye_image);
            }

            contrast_goodness_a = contrast_goodness.a;
            eye_contrast_goodness_v = contrast_goodness.v.segment<3>(0);
            pupil_contrast_goodness_v = contrast_goodness.v.segment<3>(3);
        } else {
            // Otherwise, calculate the gradient wrt. to the 3 pupil params
            typedef ::ceres::Jet<double,3> PupilJet;

            Eigen::Matrix<PupilJet,3,1> eye_pos(PupilJet(init_eye.centre[0]), PupilJet(init_eye.centre[1]), PupilJet(init_eye.centre[2]));
            ::Sphere<PupilJet> eye(eye_pos, PupilJet(init_eye.radius));

            PupilJet contrast_goodness;
            {
                EllipseGoodnessFunction<PupilJet> goodnessFunction;
                auto theta = PupilJet(pupil_vars[0], 0);
                auto psi = PupilJet(pupil_vars[1], 1);
                auto r = PupilJet(pupil_vars[2], 2);
                contrast_goodness = goodnessFunction(eye,
                    theta, psi, r,
                    PupilJet(focal_length),
                    band_width, step_epsilon,
                    eye_image);
            }

            contrast_goodness_a = contrast_goodness.a;
            pupil_contrast_goodness_v = contrast_goodness.v;
        }

        double goodness;
        auto& eye_gradient = (*gradient)[eye_var_idx()];
        auto& pupil_gradient = (*gradient)[pupil_var_idx()];

        // No smoothness term, goodness and gradient are based only on frame goodness
        goodness = contrast_goodness_a;
        if (has_eye_var)
            eye_gradient = eye_contrast_goodness_v;
        pupil_gradient = pupil_contrast_goodness_v;

        // Flip sign to change goodness into cost (i.e. maximising into minimising)
        auto cost = -goodness;
        for (int i = 0; i < number_of_variables(); ++i) {
            (*gradient)[i] = -(*gradient)[i];
        }
        return cost;
    }
    virtual double evaluate(double * const * const variables,
        std::vector<Eigen::VectorXd>* gradient,
        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const override {
            throw std::runtime_error("Not implemented");
    }

};

// Anthropomorphic term
struct PupilAnthroTerm : public spii::Term {
    double mean;
    double sigma;
    double scale;

    PupilAnthroTerm(double mean, double sigma, double scale) : mean(mean), sigma(sigma), scale(scale)
    {}

    virtual int number_of_variables() const override {
        int nvars = 1; // This pupil params
        return nvars;
    }
    virtual int variable_dimension(int var) const override {
        if (var == 0) // This pupil params (r)
            return 3;
        return -1;
    }
    virtual double evaluate(double * const * const vars) const override
    {
        using math::sq;

        auto r = vars[0][2];
        auto radius_anthro_goodness = exp(-sq(r - mean)/sq(sigma));

        double goodness = radius_anthro_goodness;

        // Flip sign to change goodness into cost (i.e. maximising into minimising)
        auto cost = -goodness*scale;
        return cost;
    }
    virtual double evaluate(double * const * const vars, std::vector<Eigen::VectorXd>* gradient) const override
    {
        using math::sq;

        auto r = ceres::Jet<double,1>(vars[0][2], 0);
        auto radius_anthro_goodness = exp(-sq(r - mean)/sq(sigma));

        double goodness = radius_anthro_goodness.a;
        (*gradient)[0].segment<1>(2) = radius_anthro_goodness.v;

        // Flip sign to change goodness into cost (i.e. maximising into minimising)
        auto cost = -goodness*scale;
        for (int i = 0; i < number_of_variables(); ++i) {
            (*gradient)[i] = -(*gradient)[i]*scale;
        }
        return cost;
    }
    virtual double evaluate(double * const * const variables,
        std::vector<Eigen::VectorXd>* gradient,
        std::vector< std::vector<Eigen::MatrixXd> >* hessian) const override {
            throw std::runtime_error("Not implemented");
    }

};

const EyeModelFitter::Vector3 EyeModelFitter::camera_centre = EyeModelFitter::Vector3::Zero();


EyeModelFitter::Pupil::Pupil(Observation observation) : observation(observation), params(0, 0, 0)
{

}

EyeModelFitter::Pupil::Pupil()
{

}


EyeModelFitter::PupilParams::PupilParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius)
{

}

EyeModelFitter::PupilParams::PupilParams() : theta(0), psi(0), radius(0)
{

}


EyeModelFitter::Observation::Observation(cv::Mat image, Ellipse ellipse, std::vector<cv::Point2f> inliers) : image(std::move(image)), ellipse(std::move(ellipse)), inliers(std::move(inliers))
{

}

EyeModelFitter::Observation::Observation()
{

}

}


singleeyefitter::EyeModelFitter::EyeModelFitter() : region_band_width(5), region_step_epsilon(0.5), region_scale(1)
{

}
singleeyefitter::EyeModelFitter::EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon) : focal_length(focal_length), region_band_width(region_band_width), region_step_epsilon(region_step_epsilon), region_scale(1)
{

}

singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_observation(cv::Mat image, Ellipse pupil, int n_pseudo_inliers /*= 0*/)
{
    std::vector<cv::Point2f> pupil_inliers;
    for (int i = 0; i < n_pseudo_inliers; ++i) {
        auto p = pointAlongEllipse(pupil, i * 2 * M_PI / n_pseudo_inliers);
        pupil_inliers.emplace_back(static_cast<float>(p[0]), static_cast<float>(p[1]));
    }
    return add_observation(std::move(image), std::move(pupil), std::move(pupil_inliers));
}

singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_observation(cv::Mat image, Ellipse pupil, std::vector<cv::Point2f> pupil_inliers)
{
    assert(image.channels() == 1 && image.depth() == CV_8U);

    std::lock_guard<std::mutex> lock_model(model_mutex);

    pupils.emplace_back(
        Observation(std::move(image), std::move(pupil), std::move(pupil_inliers))
        );
    return pupils.size() - 1;
}

void EyeModelFitter::reset()
{
    std::lock_guard<std::mutex> lock_model(model_mutex);
    pupils.clear();
    eye = Sphere::Null;
    model_version++;
}

singleeyefitter::EyeModelFitter::Circle singleeyefitter::EyeModelFitter::circleFromParams(const Sphere& eye, const PupilParams& params)
{
    if (params.radius == 0)
        return Circle::Null;

    Vector3 radial = sph2cart<double>(double(1), params.theta, params.psi);
    return Circle(eye.centre + eye.radius * radial,
        radial,
        params.radius);
}

singleeyefitter::EyeModelFitter::Circle singleeyefitter::EyeModelFitter::circleFromParams(const PupilParams& params) const
{
    return circleFromParams(eye, params);
}

void singleeyefitter::EyeModelFitter::print_single_contrast_metric(const Pupil& pupil) const
{
    if (!pupil.circle) {
        std::cout << "No pupil" << std::endl;
        return;
    }

    double params[3];
    params[0] = pupil.params.theta;
    params[1] = pupil.params.psi;
    params[2] = pupil.params.radius;
    double* vars[1];
    vars[0] = params;

    std::vector<Eigen::VectorXd> gradient;
    gradient.push_back(Eigen::VectorXd::Zero(3));

    PupilContrastTerm<false> contrast_term(
        eye,
        focal_length * region_scale,
        cvx::resize(pupil.observation.image, region_scale),
        region_band_width,
        region_step_epsilon);

    double contrast_val = contrast_term.evaluate(vars, &gradient);

    std::cout << "Contrast term: " << contrast_val << std::endl;
    std::cout << "     gradient: [ " << gradient[0].transpose() << " ]" << std::endl;
}

void singleeyefitter::EyeModelFitter::print_single_contrast_metric(Index id) const
{
    print_single_contrast_metric(pupils[id]);
}

double singleeyefitter::EyeModelFitter::single_contrast_metric(const Pupil& pupil) const
{
    if (!pupil.circle) {
        std::cout << "No pupil" << std::endl;
        return 0;
    }

    double params[3];
    params[0] = pupil.params.theta;
    params[1] = pupil.params.psi;
    params[2] = pupil.params.radius;
    double* vars[1];
    vars[0] = params;

    PupilContrastTerm<false> contrast_term(
        eye,
        focal_length * region_scale,
        cvx::resize(pupil.observation.image, region_scale),
        region_band_width,
        region_step_epsilon);

    double contrast_val = contrast_term.evaluate(vars);

    return contrast_val;
}

double singleeyefitter::EyeModelFitter::single_contrast_metric(Index id) const
{
    return single_contrast_metric(pupils[id]);
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::refine_single_with_contrast(Pupil& pupil)
{
    if (!pupil.circle)
        return pupil.circle;

    double params[3];
    params[0] = pupil.params.theta;
    params[1] = pupil.params.psi;
    params[2] = pupil.params.radius;

    spii::Function f;
    f.add_variable(&params[0], 3);
    f.add_term(std::make_shared<PupilContrastTerm<false>>(
        eye,
        focal_length * region_scale,
        cvx::resize(pupil.observation.image, region_scale),
        region_band_width,
        region_step_epsilon), &params[0]);

    spii::LBFGSSolver solver;
    solver.log_function = [](const std::string&) {};
    //solver.function_improvement_tolerance = 1e-5;
    spii::SolverResults results;
    solver.solve(f, &results);
    //std::cout << results << std::endl;

    pupil.params = PupilParams(params[0], params[1], params[2]);
    pupil.circle = circleFromParams(pupil.params);

    return pupil.circle;
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::refine_single_with_contrast(Index id)
{
    return refine_single_with_contrast(pupils[id]);
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::initialise_single_observation(Pupil& pupil)
{
    // Ignore the pupil circle normal, and intersect the pupil circle
    // centre projection line with the eyeball sphere
    try {
        auto pupil_centre_sphere_intersect = intersect(Line3(camera_centre, pupil.circle.centre.normalized()),
            eye);
        auto new_pupil_centre = pupil_centre_sphere_intersect.first;

        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = pupil.circle.radius / pupil.circle.centre.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_centre.z();

        // Parametrise this new pupil position using spherical coordinates
        Vector3 centre_to_pupil = new_pupil_centre - eye.centre;
        double r = centre_to_pupil.norm();
        pupil.params.theta = acos(centre_to_pupil[1] / r);
        pupil.params.psi = atan2(centre_to_pupil[2], centre_to_pupil[0]);
        pupil.params.radius = new_pupil_radius;

        // Update pupil circle to match parameters
        pupil.circle = circleFromParams(pupil.params);
    }
    catch (no_intersection_exception&) {
        pupil.circle = Circle::Null;
        pupil.params.theta = 0;
        pupil.params.psi = 0;
        pupil.params.radius = 0;
    }

    return pupil.circle;
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::initialise_single_observation(Index id)
{
    initialise_single_observation(pupils[id]);

    /*if (id > 0 && pupils[id-1].circle) {
    // Try previous circle in case of bad fits
    EllipseGoodnessFunction<double> goodnessFunction;
    auto& pupil = pupils[id];
    auto& prevPupil = pupils[id-1];

    double currentGoodness, prevGoodness;
    if (pupil.circle) {
    currentGoodness = goodnessFunction(eye, pupil.params.theta, pupil.params.psi, pupil.params.radius, focal_length, pupil.observation.image);
    prevGoodness = goodnessFunction(eye, prevPupil.params.theta, prevPupil.params.psi, prevPupil.params.radius, focal_length, pupil.observation.image);
    }

    if (!pupil.circle || prevGoodness > currentGoodness) {
    pupil.circle = prevPupil.circle;
    pupil.params = prevPupil.params;
    }
    }*/

    return pupils[id].circle;
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::unproject_single_observation(Pupil& pupil, double pupil_radius /*= 1*/) const
{
    if (eye == Sphere::Null) {
        throw std::runtime_error("Need to get eye centre estimate first (by unprojecting multiple observations)");
    }

    // Single pupil version of "unproject_observations"

    auto unprojection_pair = unproject(pupil.observation.ellipse, pupil_radius, focal_length);

    const Vector3& c = unprojection_pair.first.centre;
    const Vector3& v = unprojection_pair.first.normal;

    Vector2 c_proj = project(c, focal_length);
    Vector2 v_proj = project(v + c, focal_length) - c_proj;

    v_proj.normalize();

    Vector2 eye_centre_proj = project(eye.centre, focal_length);

    if ((c_proj - eye_centre_proj).dot(v_proj) >= 0) {
        pupil.circle = std::move(unprojection_pair.first);
    }
    else {
        pupil.circle = std::move(unprojection_pair.second);
    }

    return pupil.circle;
}

const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::unproject_single_observation(Index id, double pupil_radius /*= 1*/)
{
    return unproject_single_observation(pupils[id], pupil_radius);
}

void singleeyefitter::EyeModelFitter::refine_with_inliers(const CallbackFunction& callback /*= CallbackFunction()*/)
{
    int current_model_version;
    Eigen::Matrix<double, Eigen::Dynamic, 1> x;
    {
        std::lock_guard<std::mutex> lock_model(model_mutex);

        current_model_version = model_version;

        x = Eigen::Matrix<double, Eigen::Dynamic, 1>(3 + 3 * pupils.size());
        x.segment<3>(0) = eye.centre;
        for (int i = 0; i < pupils.size(); ++i) {
            const PupilParams& pupil_params = pupils[i].params;
            x.segment<3>(3 + 3 * i)[0] = pupil_params.theta;
            x.segment<3>(3 + 3 * i)[1] = pupil_params.psi;
            x.segment<3>(3 + 3 * i)[2] = pupil_params.radius;
        }
    }

    ceres::Problem problem;
    for (int i = 0; i < pupils.size(); ++i) {
        const cv::Mat& eye_image = pupils[i].observation.image;
        const auto& pupil_inliers = pupils[i].observation.inliers;

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<EllipseDistanceResidualFunction<double>, ceres::DYNAMIC, 3, 3>(
            new EllipseDistanceResidualFunction<double>(eye_image, pupil_inliers, eye.radius, focal_length),
            pupil_inliers.size()
            ),
            NULL, &x[0], &x[3 + 3 * i]);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1000;
    options.function_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = true;
    options.update_state_every_iteration = true;
    if (callback) {
        struct CallCallbackWrapper : public ceres::IterationCallback
        {
            double eye_radius;
            const CallbackFunction& callback;
            const Eigen::Matrix<double, Eigen::Dynamic, 1>& x;

            CallCallbackWrapper(const EyeModelFitter& fitter, const CallbackFunction& callback, const Eigen::Matrix<double, Eigen::Dynamic, 1>& x)
                : eye_radius(fitter.eye.radius), callback(callback), x(x) {}

            virtual ceres::CallbackReturnType operator() (const ceres::IterationSummary& summary) {
                Eigen::Matrix<double, 3, 1> eye_pos(x[0], x[1], x[2]);
                Sphere eye(eye_pos, eye_radius);

                std::vector<Circle> pupils;
                for (int i = 0; i < (x.size() - 3)/3; ++i) {
                    auto&& pupil_param_v = x.segment<3>(3 + 3 * i);
                    pupils.push_back(EyeModelFitter::circleFromParams(eye, PupilParams(pupil_param_v[0], pupil_param_v[1], pupil_param_v[2])));
                }

                callback(eye, pupils);

                return ceres::SOLVER_CONTINUE;
            }
        };
        options.callbacks.push_back(new CallCallbackWrapper(*this, callback, x));
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    {
        std::lock_guard<std::mutex> lock_model(model_mutex);

        if (current_model_version != model_version)    {
            std::cout << "Old model, not applying refined parameters" << std::endl;
            return;
        }

        eye.centre = x.segment<3>(0);

        for (int i = 0; i < pupils.size(); ++i) {
            auto&& pupil_param = x.segment<3>(3 + 3 * i);
            pupils[i].params = PupilParams(pupil_param[0], pupil_param[1], pupil_param[2]);
            pupils[i].circle = circleFromParams(eye, pupils[i].params);
        }
    }
}

void singleeyefitter::EyeModelFitter::refine_with_region_contrast(const CallbackFunction& callback /*= CallbackFunction()*/)
{
    int current_model_version;
    Eigen::Matrix<double, Eigen::Dynamic, 1> x0;
    spii::Function f;
    {
        std::lock_guard<std::mutex> lock_model(model_mutex);

        current_model_version = model_version;

        x0 = Eigen::Matrix<double, Eigen::Dynamic, 1>(3 + 3 * pupils.size());
        x0.segment<3>(0) = eye.centre;
        for (int i = 0; i < pupils.size(); ++i) {
            const PupilParams& pupil_params = pupils[i].params;
            x0.segment<3>(3 + 3 * i)[0] = pupil_params.theta;
            x0.segment<3>(3 + 3 * i)[1] = pupil_params.psi;
            x0.segment<3>(3 + 3 * i)[2] = pupil_params.radius;
        }

        f.add_variable(&x0[0], 3);
        for (int i = 0; i < pupils.size(); ++i) {
            if (pupils[i].circle) {
                f.add_variable(&x0[3 + 3 * i], 3);

                f.add_term(
                    std::make_shared<PupilContrastTerm<true>>(
                    eye,
                    focal_length * region_scale,
                    cvx::resize(pupils[i].observation.image, region_scale),
                    region_band_width,
                    region_step_epsilon),
                    &x0[0], &x0[3 + 3 * i]);
                //f.add_term(std::make_shared<PupilAnthroTerm>(2.5, 1, 0.001), &x0[3+3*i]);

                /*if (i == 0 || !pupils[i-1].circle) {
                } else {
                vars.push_back(&x0[3+3*(i-1)]);
                f.add_term(std::make_shared<SinglePupilTerm<true,true,true>>(*this, pupils[i].observation.image), vars);
                }*/
            }
        }
    }

    spii::LBFGSSolver solver;
    solver.maximum_iterations = 1000;
    solver.function_improvement_tolerance = 1e-5;
    spii::SolverResults results;
    if (callback) {
        double eye_radius = eye.radius;
        solver.callback_function = [eye_radius, &callback](const spii::CallbackInformation& info) {
            auto& x = *info.x;

            Eigen::Matrix<double, 3, 1> eye_pos(x(0), x(1), x(2));
            Sphere eye(eye_pos, eye_radius);

            std::vector<Circle> pupils;
            for (int i = 0; i < (x.size() - 3) / 3; ++i) {
                pupils.push_back(EyeModelFitter::circleFromParams(eye, PupilParams(x(3 + 3 * i + 0), x(3 + 3 * i + 1), x(3 + 3 * i + 2))));
            }

            callback(eye, pupils);

            return true;
        };
    }
    solver.solve(f, &results);
    std::cout << results << std::endl;

    {
        std::lock_guard<std::mutex> lock_model(model_mutex);

        if (current_model_version != model_version)    {
            std::cout << "Old model, not applying refined parameters" << std::endl;
            return;
        }

        eye.centre = x0.segment<3>(0);

        for (int i = 0; i < (x0.size() - 3) / 3; ++i) {
            auto pupil_param = x0.segment<3>(3 + 3 * i);
            pupils[i].params = PupilParams(pupil_param[0], pupil_param[1], pupil_param[2]);
            pupils[i].circle = circleFromParams(pupils[i].params);
        }
    }
}

void singleeyefitter::EyeModelFitter::initialise_model()
{
    std::lock_guard<std::mutex> lock_model(model_mutex);

    if (eye == Sphere::Null) {
        return;
    }

    // Find pupil positions on eyeball to get radius
    //
    // For each image, calculate the 'most likely' position of the pupil
    // circle given the eyeball sphere estimate and gaze vector. Re-estimate
    // the gaze vector to be consistent with this position.

    // First estimate of pupil centre, used only to get an estimate of eye radius

    double eye_radius_acc = 0;
    int eye_radius_count = 0;

    for (const auto& pupil : pupils) {
        if (!pupil.circle) {
            continue;
        }
        if (!pupil.init_valid) {
            continue;
        }

        // Intersect the gaze from the eye centre with the pupil circle
        // centre projection line (with perfect estimates of gaze, eye
        // centre and pupil circle centre, these should intersect,
        // otherwise find the nearest point to both lines)

        Vector3 pupil_centre = nearest_intersect(Line3(eye.centre, pupil.circle.normal),
            Line3(camera_centre, pupil.circle.centre.normalized()));

        auto distance = (pupil_centre - eye.centre).norm();

        eye_radius_acc += distance;
        ++eye_radius_count;
    }

    // Set the eye radius as the mean distance from pupil centres to eye centre
    eye.radius = eye_radius_acc / eye_radius_count;

    // Second estimate of pupil radius, used to get position of pupil on eye

    for (auto& pupil : pupils) {
        initialise_single_observation(pupil);
    }

    // Scale eye to anthropomorphic average radius of 12mm
    auto scale = 12.0 / eye.radius;
    eye.radius = 12.0;
    eye.centre *= scale;
    for (auto& pupil : pupils) {
        pupil.params.radius *= scale;
        pupil.circle = circleFromParams(pupil.params);
    }

    model_version++;

    // Try previous circle in case of bad fits
    /*EllipseGoodnessFunction<double> goodnessFunction;
    for (int i = 1; i < pupils.size(); ++i) {
    auto& pupil = pupils[i];
    auto& prevPupil = pupils[i-1];

    if (prevPupil.circle) {
    double currentGoodness, prevGoodness;
    if (pupil.circle) {
    currentGoodness = goodnessFunction(eye, pupil.params.theta, pupil.params.psi, pupil.params.radius, focal_length, pupil.observation.image);
    prevGoodness = goodnessFunction(eye, prevPupil.params.theta, prevPupil.params.psi, prevPupil.params.radius, focal_length, pupil.observation.image);
    }

    if (!pupil.circle || prevGoodness > currentGoodness) {
    pupil.circle = prevPupil.circle;
    pupil.params = prevPupil.params;
    }
    }
    }*/
}

void singleeyefitter::EyeModelFitter::unproject_observations(double pupil_radius /*= 1*/, double eye_z /*= 20*/, bool use_ransac /*= true*/)
{
    using math::sq;

    std::lock_guard<std::mutex> lock_model(model_mutex);

    if (pupils.size() < 2) {
        throw std::runtime_error("Need at least two observations");
    }

    std::vector<std::pair<Circle, Circle>> pupil_unprojection_pairs;
    std::vector<Line> pupil_gazelines_proj;

    for (const auto& pupil : pupils) {
        // Get pupil circles (up to depth)
        //
        // Do a per-image unprojection of the pupil ellipse into the two fixed
        // size circles that would project onto it. The size of the circles
        // doesn't matter here, only their centre and normal does.
        auto unprojection_pair = unproject(pupil.observation.ellipse,
            pupil_radius, focal_length);

        // Get projected circles and gaze vectors
        //
        // Project the circle centres and gaze vectors down back onto the image
        // plane. We're only using them as line parametrisations, so it doesn't
        // matter which of the two centres/gaze vectors we use, as the
        // two gazes are parallel and the centres are co-linear.

        const auto& c = unprojection_pair.first.centre;
        const auto& v = unprojection_pair.first.normal;

        Vector2 c_proj = project(c, focal_length);
        Vector2 v_proj = project(v + c, focal_length) - c_proj;

        v_proj.normalize();

        pupil_unprojection_pairs.push_back(std::move(unprojection_pair));
        pupil_gazelines_proj.emplace_back(c_proj, v_proj);
    }


    // Get eyeball centre
    //
    // Find a least-squares 'intersection' (point nearest to all lines) of
    // the projected 2D gaze vectors. Then, unproject that circle onto a
    // point a fixed distance away.
    //
    // For robustness, use RANSAC to eliminate stray gaze lines
    //
    // (This has to be done here because it's used by the pupil circle
    // disambiguation)

    Vector2 eye_centre_proj;
    bool valid_eye;

    if (use_ransac) {
        auto indices = fun::range_<std::vector<size_t>>(pupil_gazelines_proj.size());

        const int n = 2;
        double w = 0.3;
        double p = 0.9999;
        int k = ceil(log(1 - p) / log(1 - pow(w, n)));

        double epsilon = 10;
        auto huber_error = [&](const Vector2& point, const Line& line) {
            double dist = euclidean_distance(point, line);
            if (sq(dist) < sq(epsilon))
                return sq(dist) / 2;
            else
                return epsilon*(abs(dist) - epsilon / 2);
        };
        auto m_error = [&](const Vector2& point, const Line& line) {
            double dist = euclidean_distance(point, line);
            if (sq(dist) < sq(epsilon))
                return sq(dist);
            else
                return sq(epsilon);
        };
        auto error = m_error;

        auto best_inlier_indices = decltype(indices)();
        Vector2 best_eye_centre_proj;// = nearest_intersect(pupil_gazelines_proj);
        double best_line_distance_error = std::numeric_limits<double>::infinity();// = fun::sum(LAMBDA(const Line& line)(error(best_eye_centre_proj,line)), pupil_gazelines_proj);

        for (int i = 0; i < k; ++i) {
            auto index_sample = singleeyefitter::randomSubset(indices, n);
            auto sample = fun::map([&](size_t i){ return pupil_gazelines_proj[i]; }, index_sample);

            auto sample_centre_proj = nearest_intersect(sample);

            auto index_inliers = fun::filter(
                [&](size_t i){ return euclidean_distance(sample_centre_proj, pupil_gazelines_proj[i]) < epsilon; },
                indices);
            auto inliers = fun::map([&](size_t i){ return pupil_gazelines_proj[i]; }, index_inliers);

            if (inliers.size() <= w*pupil_gazelines_proj.size()) {
                continue;
            }

            auto inlier_centre_proj = nearest_intersect(inliers);

            double line_distance_error = fun::sum(
                [&](size_t i){ return error(inlier_centre_proj, pupil_gazelines_proj[i]); },
                indices);

            if (line_distance_error < best_line_distance_error) {
                best_eye_centre_proj = inlier_centre_proj;
                best_line_distance_error = line_distance_error;
                best_inlier_indices = std::move(index_inliers);
            }
        }

        std::cout << "Inliers: " << best_inlier_indices.size()
            << " (" << (100.0*best_inlier_indices.size() / pupil_gazelines_proj.size()) << "%)"
            << " = " << best_line_distance_error
            << std::endl;

        for (auto& pupil : pupils) {
            pupil.init_valid = false;
        }
        for (auto& i : best_inlier_indices) {
            pupils[i].init_valid = true;
        }

        if (best_inlier_indices.size() > 0) {
            eye_centre_proj = best_eye_centre_proj;
            valid_eye = true;
        }
        else {
            valid_eye = false;
        }
    }
    else {
        for (auto& pupil : pupils) {
            pupil.init_valid = true;
        }
        eye_centre_proj = nearest_intersect(pupil_gazelines_proj);
        valid_eye = true;
    }

    if (valid_eye) {
        eye.centre << eye_centre_proj * eye_z / focal_length,
            eye_z;
        eye.radius = 1;

        // Disambiguate pupil circles using projected eyeball centre
        //
        // Assume that the gaze vector points away from the eye centre, and
        // so projected gaze points away from projected eye centre. Pick the
        // solution which satisfies this assumption
        for (size_t i = 0; i < pupils.size(); ++i) {
            const auto& pupil_pair = pupil_unprojection_pairs[i];
            const auto& line = pupil_gazelines_proj[i];

            const auto& c_proj = line.origin();
            const auto& v_proj = line.direction();

            // Check if v_proj going away from est eye centre. If it is, then
            // the first circle was correct. Otherwise, take the second one.
            // The two normals will point in opposite directions, so only need
            // to check one.
            if ((c_proj - eye_centre_proj).dot(v_proj) >= 0) {
                pupils[i].circle = std::move(pupil_pair.first);
            }
            else {
                pupils[i].circle = std::move(pupil_pair.second);
            }
        }
    }
    else {
        // No inliers, so no eye
        eye = Sphere::Null;

        // Arbitrarily pick first circle
        for (size_t i = 0; i < pupils.size(); ++i) {
            const auto& pupil_pair = pupil_unprojection_pairs[i];
            pupils[i].circle = std::move(pupil_pair.first);
        }
    }

    model_version++;
}
