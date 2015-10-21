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
#include "../common/types.h"
#include "../mathHelper.h"

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

template<typename T>
T dist(T phi1, T theta1, T phi2, T theta2 )
{
    // using std::sin;
    // using std::cos;
    // using std::acos;
    // using std::asin;
    // using std::atan2;
    // using std::sqrt;
    // using singleeyefitter::math::sq;

    // if (phi1 == phi2 && theta1 == theta2) {
    //     return T(0);
    // }
    // Haversine distance
    auto dist = T(2) * asin(sqrt( (sin((phi2 - phi1) / T(2))*sin((phi2 - phi1) / T(2))) + cos(phi1) * cos(phi2) * (sin((theta2 - theta1) / T(2))*sin((theta2 - theta1) / T(2))) ));
    return dist;


}


class DistanceFromCircleCost {
 public:
  DistanceFromCircleCost(double phi, double theta) : xx_(phi), yy_(theta) {}
  template <typename T> bool operator()(const T* const x,
                  const T* const y,
                  //const T* const r,  // r = m^2
                  T* residual) const {


    T xp = *x; //phi
    T yp = *y; //theta


    T distance = dist(xp , yp , T(xx_) ,T(yy_) );
    residual[0]  =  distance*distance;
    return true;
  }
 private:
  // The measured x,y coordinate that should be on the circle.
  double xx_, yy_;
};

// class DistanceFromCircleCost {
//  public:
//   DistanceFromCircleCost( double x, double y , double z) : x(x), y(y), z(z) {}
//   template <typename T> bool operator()(const T* const p,
//                   T* residual) const {

//     Eigen::Matrix<T,3,1> pp;
//     pp << T(x), T(y), T(z);

//     Eigen::Matrix<T,3,1> point = Eigen::Map<const Eigen::Matrix<T,3,1>>(p);
//     T dotP = pp.dot(point);
//     Eigen::Matrix<T,3,1> crossP = pp.cross(point);
//     T crossPNorm = crossP.norm();
//     T distance = atan( crossPNorm / dotP);

//     // std::cout << "pp:" <<pp[0]<<pp[1]<<pp[2]  << std::endl;
//     // std::cout << "point:" <<point[0]<<point[1]<<point[2]  << std::endl;
//     // std::cout << "crossP:" <<crossP[0]<<crossP[1]<<crossP[2]  << std::endl;
//     // std::cout << "dotP:" <<dotP << std::endl;
//     // std::cout << "crossPNorm:" <<crossPNorm << std::endl;
//     // std::cout << "distance:" <<distance << std::endl;
//     // std::cout << "atanof:" <<atanof << std::endl;



//   //  residual[0]  =  abs(r*r - distance*distance);
//     //std::cout << "xp: " << xp << " yp: " << yp  << std::endl;
//     //std::cout << "xx: " << xx_ << " yy: " << yy_  << std::endl;
// //    std::cout << "distance: " << distance << std::endl;
//     //std::cout << "residual: " << residual[0] << std::endl;
//     residual[0]  = pp.norm() - distance;
//     return true;
//   }
//  private:
//   // The measured x,y coordinate that should be on the circle.
//   double x,y,z;
// };



singleeyefitter::Vector3 find_circle( std::vector<singleeyefitter::Vector3>&  points_on_sphere, singleeyefitter::Vector3& initial_guess  ){



  Problem problem;
  double x,y,r ,initial_x, initial_y, initial_r;

  // initial guess
  initial_x = x  = initial_guess[0];
  initial_y = y  = initial_guess[1];
  initial_r = r  = initial_guess[2];

  for( auto& p : points_on_sphere){

    //calculate spehrical coords
    double xp = p[0];
    double yp = p[1];
    double zp = p[2];
    double phi, theta;

    theta = std::atan(xp/zp);
    phi = std::acos( yp / std::sqrt(xp*xp + yp*yp + zp*zp) );
    std::cout << "phi: " << phi << " theta: " << theta << std::endl;
    CostFunction *cost = new AutoDiffCostFunction<DistanceFromCircleCost , 1, 1, 1 >( new DistanceFromCircleCost(phi,theta ) );
    problem.AddResidualBlock(cost, nullptr, &x, &y );
  }
   problem.SetParameterLowerBound(&x , 0, 0);
   problem.SetParameterLowerBound(&y , 0, 0);
 //  problem.SetParameterLowerBound(&r , 0, 0.0001);
   problem.SetParameterUpperBound(&x , 0, M_PI);
   problem.SetParameterUpperBound(&y , 0, M_PI/2.0 );
 //  problem.SetParameterUpperBound(&r , 0, M_PI/2.0);

   // Build and solve the problem.
  Solver::Options options;
  options.max_num_iterations = 500;
  options.linear_solver_type = ceres::DENSE_QR;
  options.parameter_tolerance = 1e-18;
  options.function_tolerance = 1e-18;
  options.gradient_tolerance = 1e-18;
  options.minimizer_type = ceres::TRUST_REGION;
  options.minimizer_progress_to_stdout = true;
  options.check_gradients = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
// // Recover r from m.

  // std::cout << summary.BriefReport() << "\n";
  std::cout << summary.FullReport() << "\n";
  std::cout << "phi : " << initial_x << " -> " << x << "\n";
  std::cout << "theta : " << initial_y << " -> " << y << "\n";
  std::cout << "r : " << initial_r << " -> " << r << "\n";

  return singleeyefitter::Vector3(x,y,r);

}
