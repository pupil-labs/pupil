#ifndef __UTILS_H__
#define __UTILS_H__

#include "geometry/Ellipse.h"
#include "geometry/Circle.h"
#include "geometry/Sphere.h"
#include "common/constants.h"
#include "common/traits.h"
#include "mathHelper.h"

#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <stdexcept>

#include <iostream>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace singleeyefitter {

    class MakeString {
        public:
            std::stringstream stream;
            operator std::string() const { return stream.str(); }

            template<class T>
            MakeString& operator<<(T const& VAR) { stream << VAR; return *this; }
    };

    inline int pow2(int n)
    {
        return 1 << n;
    }

    int random(int min, int max);
    int random(int min, int max, unsigned int seed);
    double random(double min, double max);
    double random(double min, double max, unsigned int seed);

    template<typename T>
    std::vector<T> randomSubset(const std::vector<T>& src, typename std::vector<T>::size_type size)
    {
        if (size > src.size())
            throw std::range_error("Subset size out of range");

        std::vector<T> ret;
        std::set<size_t> vals;

        for (size_t j = src.size() - size; j < src.size(); ++j) {
            size_t idx = random(0, j); // generate a random integer in range [0, j]

            if (vals.find(idx) != vals.end())
                idx = j;

            ret.push_back(src[idx]);
            vals.insert(idx);
        }

        return ret;
    }

    template<typename T>
    std::vector<T> randomSubset(const std::vector<T>& src, typename std::vector<T>::size_type size, unsigned int seed)
    {
        if (size > src.size())
            throw std::range_error("Subset size out of range");

        std::vector<T> ret;
        std::set<size_t> vals;

        for (size_t j = src.size() - size; j < src.size(); ++j) {
            size_t idx = random(0, j, seed + j); // generate a random integer in range [0, j]

            if (vals.find(idx) != vals.end())
                idx = j;

            ret.push_back(src[idx]);
            vals.insert(idx);
        }

        return ret;
    }

    template<typename Scalar>
    inline Eigen::Matrix<Scalar, 2, 1> toEigen(const cv::Point2f& point)
    {
        return Eigen::Matrix<Scalar, 2, 1>(static_cast<Scalar>(point.x),
                                           static_cast<Scalar>(point.y));
    }
    template<typename Scalar>
    inline cv::Point2f toPoint2f(const Eigen::Matrix<Scalar, 2, 1>& point)
    {
        return cv::Point2f(static_cast<float>(point[0]),
                           static_cast<float>(point[1]));
    }
    template<typename Scalar>
    inline cv::Point toPoint(const Eigen::Matrix<Scalar, 2, 1>& point)
    {
        return cv::Point(static_cast<int>(point[0]),
                         static_cast<int>(point[1]));
    }
    template<typename Scalar>
    inline cv::Mat toMat(const Eigen::Matrix<Scalar, 3, 1>& point)
    {
        return (cv::Mat_<Scalar>(3,1) << point[0],
                                         point[1],
                                         point[2]);
    }
    template<typename Scalar>
    inline cv::Mat toMat(const Eigen::Matrix<Scalar, 2, 1>& point)
    {
        return (cv::Mat_<Scalar>(2,1) << point[0],
                                         point[1]);
    }
    template<typename Scalar>
    inline cv::RotatedRect toRotatedRect(const Ellipse2D<Scalar>& ellipse)
    {
        return cv::RotatedRect(toPoint2f(ellipse.center),
                               cv::Size2f(static_cast<float>(2.0 * ellipse.major_radius),
                                          static_cast<float>(2.0 * ellipse.minor_radius)),
                               static_cast<float>(ellipse.angle * 180.0 / constants::PI));
    }
    template<typename Scalar>
    inline Ellipse2D<Scalar> toEllipse(const cv::RotatedRect& rect)
    {
        // Scalar major = rect.size.height;
        // Scalar minor = rect.size.width;
        // if(major < minor ){
        //     std::cout << "Flip major minor !!" << std::endl;
        //     std::swap(major,minor);
        // }
        return Ellipse2D<Scalar>(toEigen<Scalar>(rect.center),
                                 static_cast<Scalar>(rect.size.height / 2.0),
                                 static_cast<Scalar>(rect.size.width / 2.0),
                                 static_cast<Scalar>((rect.angle + 90.0) * constants::PI / 180.0));
    }

        template<typename Scalar>
    cv::Rect bounding_box(const Ellipse2D<Scalar>& ellipse)
    {
        using std::sin;
        using std::cos;
        using std::sqrt;
        using std::floor;
        using std::ceil;
        Scalar ux = ellipse.major_radius * cos(ellipse.angle);
        Scalar uy = ellipse.major_radius * sin(ellipse.angle);
        Scalar vx = ellipse.minor_radius * cos(ellipse.angle + constants::PI / 2);
        Scalar vy = ellipse.minor_radius * sin(ellipse.angle + constants::PI / 2);
        Scalar bbox_halfwidth = sqrt(ux * ux + vx * vx);
        Scalar bbox_halfheight = sqrt(uy * uy + vy * vy);
        return cv::Rect(floor(ellipse.center[0] - bbox_halfwidth), floor(ellipse.center[1] - bbox_halfheight),
                        2 * ceil(bbox_halfwidth) + 1, 2 * ceil(bbox_halfheight) + 1);
    }

    template<template<class, int> class Jet, class T, int N>
    typename std::enable_if<std::is_same<typename ad_traits<Jet<T, N>>::ad_tag, ceres_jet_tag>::value, Ellipse2D<T>>::type
            toConst(const Ellipse2D<Jet<T, N>>& ellipse)
    {
        return Ellipse2D<T>(
                   ellipse.center[0].a,
                   ellipse.center[1].a,
                   ellipse.major_radius.a,
                   ellipse.minor_radius.a,
                   ellipse.angle.a);
    }

    template<class T>
    Ellipse2D<T> scaledMajorRadius(const Ellipse2D<T>& ellipse, const T& target_radius)
    {
        return Ellipse2D<T>(
                   ellipse.center[0],
                   ellipse.center[1],
                   target_radius,
                   target_radius * ellipse.minor_radius / ellipse.major_radius,
                   ellipse.angle);
    };


    template<typename T>
    T angleDiffGoodness(T theta1, T psi1, T theta2, T psi2, typename ad_traits<T>::scalar sigma)
    {
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
        auto dist = T(2) * asin(sqrt(sq(sin((theta1 - theta2) / T(2))) + cos(theta1) * cos(theta2) * sq(sin((psi1 - psi2) / T(2)))));
        return exp(-sq(dist) / sq(sigma));
    }

    template<typename T>
    Circle3D<T> circleOnSphere(const Sphere<T>& sphere, T theta, T psi, T circle_radius)
    {
        typedef Eigen::Matrix<T, 3, 1> Vector3;
        Vector3 radial = math::sph2cart<T>(T(1), theta, psi);
        return Circle3D<T>(sphere.center + sphere.radius * radial,
                           radial,
                           circle_radius);
    }
    template<typename T>
    Eigen::Matrix<T, 2, 1> paramsOnSphere(const Sphere<T>& sphere, Circle3D<T> circle )
    {
        typedef Eigen::Matrix<T, 3, 1> Vector3;
        Vector3 centerOnSphere = circle.center - sphere.center;
        return math::cart2sph<T>(centerOnSphere);

    }

} //namespace singleeyefitter

#endif // __UTILS_H__
