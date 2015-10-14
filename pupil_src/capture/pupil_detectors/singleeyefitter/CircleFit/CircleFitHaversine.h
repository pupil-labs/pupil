// This fits circles to a collection of points, where the error is related to
// the distance of a point from the circle. This uses auto-differentiation to
// take the derivatives.
//
// The input format is simple text. Feed on standard in:
//
//   x_initial y_initial r_initial
//   x1 y1
//   x2 y2
//   y3 y3
//   ...
//
// And the result after solving will be printed to stdout:
//
//   x y r
//
// There are closed form solutions [1] to this problem which you may want to
// consider instead of using this one. If you already have a decent guess, Ceres
// can squeeze down the last bit of error.
//
//   [1] http://www.mathworks.com/matlabcentral/fileexchange/5557-circle-fit/content/circfit.m
#include <cstdio>
#include <vector>
#include <ceres/ceres.h>
#include <boost/geometry/strategies/spherical/distance_haversine.hpp>
#include <boost/geometry.hpp>
#include "../common/types.h"


namespace bg = boost::geometry;

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
// DEFINE_double(robust_threshold, 0.0, "Robust loss parameter. Set to 0 for "
//               "normal squared error (no robustification).");
// The cost for a single sample. The returned residual is related to the
// distance of the point from the circle (passed in as x, y, m parameters).
//
// Note that the radius is parameterized as r = m^2 to constrain the radius to
// positive values.

class DistanceFromCircleCost {
 public:
  DistanceFromCircleCost(double xx, double yy) : xx_(xx), yy_(yy) {}
  bool operator()(const double* const x,
                  const double* const y,
                  const double* const r,  // r = m^2
                  double* residual) const {

    double sphere_radius = 1.0;
    boost::geometry::strategy::distance::haversine<double> const haversine(sphere_radius);

    // Since the radius is parameterized as m^2, unpack m to get r.
    //double r = *m * *m;
    // Get the position of the sample in the circle's coordinate system.
    //T xp = xx_ - *x;
    //T yp = yy_ - *y;
    // It is tempting to use the following cost:
    //
    //   residual[0] = r - sqrt(xp*xp + yp*yp);
    //
    // which is the distance of the sample from the circle. This works
    // reasonably well, but the sqrt() adds strong nonlinearities to the cost
    // function. Instead, a different cost is used, which while not strictly a
    // distance in the metric sense (it has units distance^2) it produces more
    // robust fits when there are outliers. This is because the cost surface is
    // more convex.
    //residual[0] = r*r - xp*xp - yp*yp;
    double xp = *x;
    double yp = *y;

    bg::model::point<double, 2, bg::cs::cartesian> point1(xx_, yy_);
    bg::model::point<double, 2, bg::cs::cartesian> point2(xp, yp);
    double distance = boost::geometry::distance(point1, point2, haversine);
    residual[0]  = *r - distance;
    return true;
  }
 private:
  // The measured x,y coordinate that should be on the circle.
  double xx_, yy_;
};



singleeyefitter::Vector3 find_circle( std::vector<singleeyefitter::Vector3>&  points_on_sphere ){

  Problem problem;
  double x,y,r ,initial_x, initial_y, initial_r;

  // initial guess
  initial_x = x  = 1.5/2.0;
  initial_y = y  = 1.5/2.0;
  initial_r = r  = 1;

  for( auto& p : points_on_sphere){

    //calculate spehrical coords
    double xp = p[0];
    double yp = p[1];
    double zp = p[2];
    double phi, theta;
    if(std::abs(xp) < 0.0000001  ) xp = 0.0;
    if(std::abs(yp) < 0.0000001  ) yp = 0.0;
    if(std::abs(zp) < 0.0000001  ) zp = 0.0;
    phi = std::atan2(zp,xp);

    theta = std::acos( yp / std::sqrt(xp*xp + yp*yp + zp*zp) ); //colatitude

    std::cout << "theta: " << theta << " phi: " << phi << std::endl;
    CostFunction *cost = new NumericDiffCostFunction<DistanceFromCircleCost, ceres::CENTRAL, 1, 1, 1, 1>( new DistanceFromCircleCost(theta, phi) );
    problem.AddResidualBlock(cost, nullptr, &x, &y, &r);
  }
   // Build and solve the problem.
  Solver::Options options;
  options.max_num_iterations = 500;
  options.linear_solver_type = ceres::DENSE_QR;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

// Recover r from m.
 // std::cout << summary.BriefReport() << "\n";
  std::cout << summary.FullReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  std::cout << "y : " << initial_y << " -> " << y << "\n";
  std::cout << "r : " << initial_r << " -> " << r << "\n";

  return singleeyefitter::Vector3(x,y,r);

}
