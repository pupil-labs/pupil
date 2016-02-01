


#include "common.h"
#include <vector>
#include <cstdio>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

struct TransformationError {
  TransformationError(const Vector3 referencePoint,   const Vector3 gazePoint1, const Vector3 gazePoint2 )
      : referencePoint(referencePoint), gazePoint1(gazePoint1), gazePoint2(gazePoint2) {}

  template <typename T>
  bool operator()(
                  const T* const R_t,  // Rotation denoted by angle axis
                                       // followed with translation
                  T* residuals) const {


    Eigen::Matrix<T, 3,1> translation = Eigen::Map<const Eigen::Matrix<T, 3,1>>(&R_t[3] , 3 );
    // Compute coordinates with current transformation matrix: x = RX + t.
    Eigen::Matrix<T, 3,1> p1;
    Eigen::Matrix<T, 3,1> p2;

    T g1[3] = {T(gazePoint1[0]),T(gazePoint1[3]),T(gazePoint1[2])};
    T g2[3] = {T(gazePoint2[0]),T(gazePoint2[3]),T(gazePoint2[2])};

    ceres::AngleAxisRotatePoint(R_t, g1 , p1.data());
    ceres::AngleAxisRotatePoint(R_t, g2, p2.data());

    p1 += translation;
    p2 += translation;

    Eigen::Matrix<T, 3,1> refP;
    refP << T(referencePoint[0]) , T(referencePoint[1]) ,T(referencePoint[2]);

    // now calculate the distance between the observed point and the nearest point on the line
   // T distance =  (p1-p2).cross(p1-refP).norm() / (refP - p2).norm()
    T dd = (refP - p2).squaredNorm();
    if(dd >= 0.000001)
        residuals[0] =  (p1-p2).cross(p1-refP).squaredNorm() / (refP - p2).squaredNorm();
    else
        residuals[0] = T(10000.0);


    return true;
  }

  const Vector3 referencePoint;
  const Vector3 gazePoint1;
  const Vector3 gazePoint2;
};


Eigen::Matrix4d pointLineCalibration( Vector3 spherePosition, const std::vector<Vector3>& refPoints, const std::vector<Vector3>& gazeDirections,  std::vector<Vector3>& gazePoints  ){



      Problem problem;
      double transformation[6] = {0.0 , -1.5 ,0.0 ,0.0 ,0.0 ,0.0 };

      int i = 0;
      for( auto& p : refPoints){
        auto g = gazeDirections.at(i);
        i++;
        CostFunction *cost = new AutoDiffCostFunction<TransformationError , 1, 6 >( new TransformationError(p , g , spherePosition ) );
        problem.AddResidualBlock(cost, nullptr, &transformation[0] );
      }


       // Build and solve the problem.
      Solver::Options options;
      options.max_num_iterations = 5000;
      options.linear_solver_type = ceres::DENSE_QR;
      options.parameter_tolerance = 1e-18;
      options.function_tolerance = 1e-18;
      options.gradient_tolerance = 1e-18;
     // options.minimizer_type = ceres::TRUST_REGION;
      options.minimizer_progress_to_stdout = true;
      options.check_gradients = true;
      Solver::Summary summary;
      Solve(options, &problem, &summary);
    // // Recover r from m.

      // std::cout << summary.BriefReport() << "\n";
      std::cout << summary.FullReport() << "\n";

    for(int i =0; i < 6; i++){
        std::cout << "," << transformation[i] << std::endl;
    }
    return Eigen::Matrix4d::Identity();
}

