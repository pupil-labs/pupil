#ifndef distance_h__
#define distance_h__

#include <vector>
#include <Eigen/Core>
#include "geometry/Ellipse.h"
#include "DistancePointEllipse.h"

using namespace singleeyefitter;

template<class Derived1, class Derived2>
auto euclidean_distance(const Eigen::MatrixBase<Derived1>& p1, const Eigen::MatrixBase<Derived2>& p2) -> decltype((p1 - p2).norm())
{
    return (p1 - p2).norm();
}
template<class Derived1, class Derived2>
auto euclidean_distance_squared(const Eigen::MatrixBase<Derived1>& p1, const Eigen::MatrixBase<Derived2>& p2) -> decltype((p1 - p2).squaredNorm())
{
    return (p1 - p2).squaredNorm();
}
template<class Derived>
auto euclidean_distance(const Eigen::MatrixBase<Derived>& point,
                        const Eigen::ParametrizedLine<typename Derived::Scalar, Derived::SizeAtCompileTime>& line) -> decltype(point.norm())
{
    return ((line.origin() - point) - ((line.origin() - point).dot(line.direction())) * line.direction()).norm();
}

template<class Derived>
auto euclidean_distance_squared(const Eigen::MatrixBase<Derived>& point,
                        const Eigen::ParametrizedLine<typename Derived::Scalar, Derived::SizeAtCompileTime>& line) -> decltype(point.norm())
{
    return ((line.origin() - point) - ((line.origin() - point).dot(line.direction())) * line.direction()).squaredNorm();
}


template<class Scalar>
Scalar euclidean_distance(const Eigen::ParametrizedLine<Scalar, 3>& line1,
                                  const Eigen::ParametrizedLine<Scalar, 3>& line2)
{
    return std::sqrt( euclidean_distance_squared(line1, line2) ) ;
}

template<class Scalar>
Scalar euclidean_distance_squared(const Eigen::ParametrizedLine<Scalar, 3>& line1,
                                  const Eigen::ParametrizedLine<Scalar, 3>& line2)
{

    auto closestPoints = closest_points_on_line(line1, line2);
    auto diff = closestPoints.first - closestPoints.second;
    return diff.dot(diff);
}

template<class Scalar>
std::pair< typename Eigen::ParametrizedLine<Scalar, 3>::VectorType , typename Eigen::ParametrizedLine<Scalar, 3>::VectorType >
closest_points_on_line(const Eigen::ParametrizedLine<Scalar, 3>& line1,
                            const Eigen::ParametrizedLine<Scalar, 3>& line2)
{
    typedef typename Eigen::ParametrizedLine<Scalar, 3>::VectorType Vector;
    Vector diff = line1.origin() - line2.origin();
    Scalar a01 = -line1.direction().dot(line2.direction());
    Scalar b0 = diff.dot(line1.direction());
    Scalar s0, s1;

    if (std::abs(a01) < Scalar(1) )
    {
        // Lines are not parallel.
        Scalar det = Scalar(1) - a01 * a01;
        Scalar b1 = -diff.dot(line2.direction());
        s0 = (a01 * b1 - b0) / det;
        s1 = (a01 * b0 - b1) / det;
    }
    else
    {
        // Lines are parallel, select any pair of closest points.
        s0 = -b0;
        s1 = Scalar(0);
    }

    Vector closestPoint1 = line1.origin() + s0 * line1.direction();
    Vector closestPoint2 = line2.origin() + s1 * line2.direction();
    return std::pair<Vector, Vector>(closestPoint1, closestPoint2);
}

template<typename Scalar, int Dim>
Scalar euclidean_distance(const Eigen::Matrix<Scalar, Dim, 1>& p, const Eigen::Matrix<Scalar, Dim, 1>& v, const Eigen::Matrix<Scalar, Dim, 1>& w)
{
    // Return minimum distance between line segment vw and point p
    auto l2 = (v - w).squaredNorm();

    if (l2 == 0.0)
        return euclidean_distance(p, v);   // v == w case

    // Consider the line extending the segment, parameterized as v + t (w - v).
    // We find projection of point p onto the line.
    // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    auto t = (p - v).dot(w - v) / l2;

    if (t < 0.0)
        return euclidean_distance(p, v);  // Beyond the 'v' end of the segment
    else if (t > 1.0)
        return euclidean_distance(p, w);  // Beyond the 'w' end of the segment

    auto projection = v + t * (w - v);  // Projection falls on the segment
    return euclidean_distance(p, projection);
}

template<typename Scalar>
Scalar euclidean_distance(const Eigen::Matrix<Scalar, 2, 1>& point, const std::vector<Eigen::Matrix<Scalar, 2, 1>>& polygon)
{
    auto from = polygon.back();
    Scalar min_distance = std::numeric_limits<Scalar>::infinity();

    for (const auto& to : polygon) {
        min_distance = std::min(min_distance, euclidean_distance(point, from, to));
        from = to;
    }

    return min_distance;
}
template<typename Scalar>
Scalar euclidean_distance( const std::vector<Eigen::Matrix<Scalar, 3, 1>>& polygon)
{
    auto from = polygon.back();
    Scalar distance = 0.0;

    for (const auto& to : polygon) {
        distance +=  euclidean_distance(from, to);
        from = to;
    }

    return distance;
}


template<typename Scalar>
Scalar euclidean_distance(const Eigen::Matrix<Scalar, 2, 1>& point, const Ellipse2D<Scalar>& ellipse)
{
    return DistancePointEllipse<Scalar>(ellipse, point[0], point[1]);
}
template<typename Scalar>
Scalar euclidean_distance(const Scalar x, const Scalar y, const Ellipse2D<Scalar>& ellipse)
{
    return DistancePointEllipse<Scalar>(ellipse, x, y);
}

template<typename Scalar, typename TOther>
Scalar oneway_hausdorff_distance(const Ellipse2D<Scalar>& ellipse, const TOther& other)
{
    Scalar max_dist = -1;

    for (Scalar i = 0; i < 100; ++i) {
        Scalar t = i * 2 * M_PI / 100;
        auto pt = pointAlongEllipse(ellipse, t);
        Scalar i_dist = euclidean_distance(pt, other);
        max_dist = std::max(max_dist, i_dist);
    }

    return max_dist;
}
template<typename Scalar, typename TOther>
Scalar oneway_hausdorff_distance(const std::vector<Eigen::Matrix<Scalar, 2, 1>>& polygon, const TOther& other)
{
    Scalar max_dist = -1;

    for (const auto& pt : polygon) {
        Scalar pt_dist = euclidean_distance(pt, other);
        max_dist = std::max(max_dist, pt_dist);
    }

    return max_dist;
}

template<typename Scalar, typename TOther>
Scalar hausdorff_distance(const Ellipse2D<Scalar>& ellipse, const TOther& other)
{
    return std::max(oneway_hausdorff_distance(ellipse, other), oneway_hausdorff_distance(ellipse, other));
}
template<typename Scalar, typename TOther>
typename std::enable_if < !std::is_same<TOther, Ellipse2D<Scalar>>::value, Scalar >::type
        hausdorff_distance(const TOther& other, const Ellipse2D<Scalar>& ellipse)
{
    return hausdorff_distance(ellipse, other);
}

#endif // distance_h__
