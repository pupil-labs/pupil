/*
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
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
#include "math/Distance.h"
#include "common/types.h"

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;



// struct CoplanarityError {
//     CoplanarityError(Vector3 refDirection, Vector3 gazeDirection, bool useWeight )
//         : refDirection(refDirection), gazeDirection(gazeDirection), useWeight(useWeight) {}

//     template <typename T>
//     bool operator()(
//         const T* const rotation,  // rotation denoted by quaternion Parameterization
//         const T* const translation,  // followed by translation
//         T* residuals) const
//     {

//         Eigen::Matrix<T, 3, 1> gazeD = {T(gazeDirection[0]), T(gazeDirection[1]), T(gazeDirection[2])};
//         Eigen::Matrix<T, 3, 1> refD = {T(refDirection[0]) , T(refDirection[1]) , T(refDirection[2])};
//         Eigen::Matrix<T, 3, 1> b = {T(translation[0]) , T(translation[1]) , T(translation[2])};

//         Eigen::Matrix<T, 3, 1> gazeWorld;

//         pupillabs::EigenQuaternionRotatePoint( rotation , gazeD.data(), gazeWorld.data() );
//         //coplanarity constraint  x1.T * E * x2 = 0
//         auto res = refD.transpose() * ( b.cross(gazeWorld));

//         const T generalVariance = T(0.8);
//         const T worldCameraVariance = T(1200);
//         const T eyeVariance = T(300);

//         if ( useWeight)
//         {
//             // weighting factor:

//             const Eigen::Matrix<T, 3, 1> distanceVector = gazeWorld.cross(refD);
//             const T distance = distanceVector.norm();
//             const T distanceSquared = distance*distance;

//             const T numerator  = distanceSquared * generalVariance;

//             const T c = b.cross(refD).dot(distanceVector);
//             const T d = b.cross(gazeWorld).dot(distanceVector);

//             const T denom = c * c * worldCameraVariance + d * d * eyeVariance;

//             const T weight = numerator / denom;

//             residuals[0] = weight * res[0]* res[0];

//         }else{
//             residuals[0] =  res[0]* res[0];
//         }

//         return true;

//     }

//     const Vector3 gazeDirection;
//     const Vector3 refDirection;
//     const bool useWeight;
// };

struct ReprojectionError {
  ReprojectionError( Vector3 observed_dir)
      : observed_dir(observed_dir) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {

        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];


        // Normalize / project back to unit sphere
        T s = sqrt( p[0]*p[0] + p[1]*p[1] + p[2]*p[2]  );
        p[0] /= s;
        p[1] /= s;
        p[2] /= s;

        // should the residual be the distance squared between the directions or better the angle ?

        // The error is the difference between the predicted and observed position.
        residuals[0] = p[0] - observed_dir[0];
        residuals[1] = p[1] - observed_dir[1];
        residuals[2] = p[2] - observed_dir[2];

        // residuals[0] = predicted_x - T(observed_x);
        // residuals[1] = predicted_y - T(observed_y);
        // residuals[1] = predicted_y - T(observed_y);
        return true;
  }

// Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Vector3 observed_dir ) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 3, 7, 3>(
                new ReprojectionError(observed_dir)));
  }

  Vector3 observed_dir;
};

bool bundleAdjustCalibration( std::vector<Observation> observations, double& avgDistance, std::vector<std::vector<double>>& cameras,  bool fixTranslation = false, bool useWeight = true )
{


    Problem problem;
    std::vector<Vector3> predicted_points;
    std::vector<double> translationLenghts;
    // calculate the predicted points form the dir of the world camera
    // world camera is the firs one in observations

    for( const auto& dir : observations.at(0).dirs ){
        predicted_points.push_back( dir * 500.0 );
    }

    for( auto& obs : observations){

        std::vector<double>& camera = obs.camera;

        double lenght = std::sqrt(  camera[4]*camera[4]+ camera[5]*camera[5] + camera[6]*camera[6]);
        camera[4] /= lenght;
        camera[5] /= lenght;
        camera[6] /= lenght;
        translationLenghts.push_back(lenght);
    }


    bool lockedCamera = false;
    for( auto& observation : observations ){

        double* camera = observation.camera.data();

        int index = 0;
        for( auto& dir : observation.dirs){

            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction* cost_function =
                ReprojectionError::Create(dir);

            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     camera,
                                     predicted_points[index].data() );
            index++;

        }

        ceres::LocalParameterization* quaternionParameterization = new pupillabs::EigenQuaternionParameterization; // owned by the problem
        problem.SetParameterization(camera, quaternionParameterization);

        if (fixTranslation)
        {
            std::vector<int> constant_translation;
            // First four elements are rotation, last three are translation.
            constant_translation.push_back(4);
            constant_translation.push_back(5);
            constant_translation.push_back(6);
            ceres::SubsetParameterization *constantTranslation = new ceres::SubsetParameterization(7, constant_translation);
            problem.SetParameterization(camera, constantTranslation);
        }else{

            ceres::LocalParameterization* normedTranslationParameterization = new pupillabs::Fixed3DNormParametrization(1.0); // owned by the problem
            problem.SetParameterization(camera+4, normedTranslationParameterization);
        }


        if(!lockedCamera){
            // first observation is world camera
            // fix translation and orientation
            problem.SetParameterBlockConstant(camera) ;
            lockedCamera = true;
        }

    }


    //get result
    for( int i=0; i < observations.size(); i++){

        double length = translationLenghts.at(i);
        std::vector<double>& camera = observations.at(i).camera;

        camera[4] *= length;
        camera[5] *= length;
        camera[6] *= length;
        cameras.push_back( observations.at(i).camera );

    }



    // double epsilon = std::numeric_limits<double>::epsilon();

    // for(int i=0; i<refDirections.size(); i++) {

    //     auto& gaze = gazeDirections.at(i);
    //     auto& ref = refDirections.at(i);
    //     gaze.normalize(); //just to be sure
    //     ref.normalize(); //just to be sure

    //     // do a check to handle parameters we can't solve
    //     // First: the length of the directions must not be zero
    //     // Second: the angle between gaze direction and reference direction must not be greater 90 degrees, considering the initial orientation
    //     bool valid = true;
    //     valid &= gaze.norm() >= epsilon;
    //     valid &= ref.norm() >= epsilon;
    //     valid &= (orientation*gaze).dot(ref) >= epsilon;

    //     if( valid ){

    //         CostFunction* cost = new AutoDiffCostFunction<CoplanarityError , 1, 4, 3 >(new CoplanarityError(ref, gaze, useWeight ));
    //         // TODO use a loss function, to handle gaze point outliers
    //         problem.AddResidualBlock(cost, nullptr, orientation.coeffs().data() ,  translation.data() );
    //     }else{
    //         std::cout << "no valid direction vector"  << std::endl;
    //     }
    // }

    // if( problem.NumResidualBlocks() == 0 ){
    //     std::cout << "nothing to solve"  << std::endl;
    //     return false;
    // }

    // ceres::LocalParameterization* quaternionParameterization = new pupillabs::EigenQuaternionParameterization; // owned by the problem
    // problem.SetParameterization(orientation.coeffs().data(), quaternionParameterization);

    // ceres::LocalParameterization* normedTranslationParameterization = new pupillabs::Fixed3DNormParametrization(1.0); // owned by the problem
    // problem.SetParameterization(translation.data(), normedTranslationParameterization);

    // if (fixTranslation)
    // {
    //     problem.SetParameterBlockConstant(translation.data());
    // }

    // Build and solve the problem.
    Solver::Options options;
    options.max_num_iterations = 3000;
    options.linear_solver_type = ceres::DENSE_QR;

    options.parameter_tolerance = 1e-25;
    options.function_tolerance = 1e-26;
    options.gradient_tolerance = 1e-30;
    //options.minimizer_progress_to_stdout = true;
    //options.logging_type = ceres::SILENT;
    //options.check_gradients = true;


    Solver::Summary summary;
    Solve(options, &problem, &summary);

    // std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    if( summary.termination_type != ceres::TerminationType::CONVERGENCE  ){
        std::cout << "Termination Error: " << ceres::TerminationTypeToString(summary.termination_type) << std::endl;
        return false;
    }

    //rescale the translation according to the initial translation
    // translation *= n;


    // using singleeyefitter::Line3;
    // // check for possible ambiguity
    // //intersection points need to lie in positive z

    // auto checkResult = [ &gazeDirections, &refDirections ]( Eigen::Quaterniond& orientation , Vector3 translation, double& avgDistance ){

    //     int validCount = 0;
    //     avgDistance = 0.0;
    //     for(int i=0; i<refDirections.size(); i++) {

    //         auto gaze = gazeDirections.at(i);
    //         auto ref = refDirections.at(i);

    //         gaze.normalize(); //just to be sure
    //         ref.normalize(); //just to be sure

    //         Vector3 gazeWorld = orientation * gaze;

    //         Line3 refLine = { Vector3(0,0,0) , ref  };
    //         Line3 gazeLine = { translation , gazeWorld  };

    //         auto closestPoints = closest_points_on_line( refLine , gazeLine );

    //         if( closestPoints.first.z() > 0.0 && closestPoints.second.z() > 0.0 )
    //             validCount++;

    //         avgDistance += euclidean_distance( closestPoints.first, closestPoints.second );
    //     }
    //     avgDistance /= refDirections.size();

    //     return validCount;
    // };


    // Eigen::Quaterniond q1 = orientation;
    // Vector3 t1 =  translation;
    // Eigen::Quaterniond q2  = q1.conjugate();
    // Vector3 t2 =  -t1;
    // double avgD1,avgD2,avgD3,avgD4;


    // std::cout << "q2: " << q2.w() << " " << q2.x() << " " << q2.y() << " " <<q2.z() << std::endl;
    // std::cout << "q1: " << q1.w() << " " << q1.x() << " " << q1.y() << " "<< q1.z() << std::endl;
    // int s1 = checkResult(q1,t1,avgD1);
    // int s2 = checkResult(q1,t2,avgD2);
    // int s3 = checkResult(q2,t1,avgD3);
    // int s4 = checkResult(q2,t2,avgD4);

    // std::cout << "s1: " << s1 << std::endl;
    // std::cout << "s2: " << s2 << std::endl;
    // std::cout << "s3: " << s3 << std::endl;
    // std::cout << "s4: " << s4 << std::endl;

    // std::vector<int> v = {s1,s2,s3,s4};
    // int maxIndex = -1;
    // int maxValue = 0;
    // int i = 0;
    // for( auto s : v ){

    //     if(s > maxValue){
    //         maxValue = s;
    //         maxIndex = i;
    //     }
    //     i++;
    // }

    // std::cout << "result one" <<std::endl;
    // avgDistance = avgD1;

    // switch(maxIndex){

    //     case 0:
    //         std::cout << "result one" <<std::endl;
    //         avgDistance = avgD1;
    //         break;
    //     case 1:

    //         std::cout << "result two" <<std::endl;
    //         avgDistance = avgD2;
    //         translation = t2;
    //         break;

    //     case 2:
    //         std::cout << "result three" <<std::endl;
    //         avgDistance = avgD3;
    //         orientation = q2;
    //         break;
    //     case 3:
    //         std::cout << "result four" <<std::endl;
    //         avgDistance = avgD4;
    //         orientation = q2;
    //         translation = t2;

    // }

    return true;

}

