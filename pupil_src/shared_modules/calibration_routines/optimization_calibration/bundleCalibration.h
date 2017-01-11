/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/


#include "common.h"
#include <vector>
#include <cstdio>
#include <limits>

#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include "ceres/Fixed3DNormParametrization.h"
#include "ceres/EigenQuaternionParameterization.h"
#include "ceres/CeresUtils.h"
#include "math/distance.h"
#include "common/types.h"

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


struct ReprojectionError {
  ReprojectionError( ::Vector3 observed_point)
      : observed_point(observed_point) {}

  template <typename T>
  bool operator()(const T* const orientation,
                  const T* const translation,
                  const T* const point,
                  T* residuals) const {

        T p[3];


        // convetional order rot and then trans
        // ceres::AngleAxisRotatePoint(orientation, point, p);
        // // pose[3,4,5] are the translation.
        // p[0] += translation[0];
        // p[1] += translation[1];
        // p[2] += translation[2];


        // unconvetional reverse order trans then rot.
        T tp[3];
        // pose[3,4,5] are the translation.
        tp[0] = point[0] + translation[0];
        tp[1] = point[1] + translation[1];
        tp[2] = point[2] + translation[2];
        ceres::AngleAxisRotatePoint(orientation, tp, p);

        // Normalize / project back to unit sphere
        T s = sqrt( p[0]*p[0] + p[1]*p[1] + p[2]*p[2]  );
        p[0] /= s;
        p[1] /= s;
        p[2] /= s;


        // The error is the difference between the predicted and observed position.
        residuals[0] = p[0] - T(observed_point[0]);
        residuals[1] = p[1] - T(observed_point[1]);
        residuals[2] = p[2] - T(observed_point[2]);

        return true;
  }

// Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const ::Vector3 observed_point ) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 3, 3, 3, 3>(
                new ReprojectionError(observed_point)));
  }

  ::Vector3 observed_point;
};

double bundleAdjustCalibration( std::vector<Observer>& observers, std::vector<::Vector3>& points,bool fix_points)
{


    Problem problem;

    for( auto& observer : observers ){

        double* pose = observer.pose.data();

        int index = 0;
        for( auto& observation : observer.observations){

            // Each Residual block takes a point and a pose as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction* cost_function =
                ReprojectionError::Create(observation);

            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     pose,
                                     pose+3,
                                     points[index].data() );
            index++;

        }
      if(observer.fix_rotation == 1){
          problem.SetParameterBlockConstant(pose);
        }
      if(observer.fix_translation == 1){
          problem.SetParameterBlockConstant(pose+3);
        }
      }

      if(fix_points == true){
        int index = 0;
        for(auto o : observers[0].observations){
            problem.SetParameterBlockConstant(points[index].data());
            index++;
            }
        }


    // Build and solve the problem.
    Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    // options.parameter_tolerance = 1e-35;
    // options.function_tolerance = 1e-35;
    options.gradient_tolerance = 1e-35;
    // options.minimizer_progress_to_stdout = true;
    //options.logging_type = ceres::SILENT;
    // options.check_gradients = true;


    Solver::Summary summary;
    Solve(options, &problem, &summary);


    std::cout << summary.BriefReport() << "\n";
    // std::cout << summary.FullReport() << "\n";

    if( summary.termination_type != ceres::TerminationType::CONVERGENCE  ){
        std::cout << "Termination Error: " << ceres::TerminationTypeToString(summary.termination_type) << std::endl;
        return -1;
    }

    return summary.final_cost;

}

