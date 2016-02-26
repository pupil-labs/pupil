


#include "common.h"
#include <vector>
#include <cstdio>
#include <limits>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include "ceres/CeresParametrization.h"
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


struct CoplanarityError {
    CoplanarityError(const Vector3 refDirection,   const Vector3 gazeDirection )
        : refDirection(refDirection), gazeDirection(gazeDirection) {}

    template <typename T>
    bool operator()(
        const T* const rotation,  // rotation denoted by quaternion Parameterization
        const T* const translation,  // followed by translation
        T* residuals) const
    {

        Eigen::Matrix<T, 3, 1> gazeD = {T(gazeDirection[0]), T(gazeDirection[1]), T(gazeDirection[2])};
        Eigen::Matrix<T, 3, 1> refD = {T(refDirection[0]) , T(refDirection[1]) , T(refDirection[2])};
        Eigen::Matrix<T, 3, 1> t = {T(translation[0]) , T(translation[1]) , T(translation[2])};

        Eigen::Matrix<T, 3, 1> gazeWorld;
        ceres::QuaternionRotatePoint( rotation , gazeD.data(), gazeWorld.data() );
        //TODO add weighting factors to the residual , better approximation
        //coplanarity constraint  x1.T * E * x2 = 0
        auto res = refD.transpose() * ( t.cross(gazeWorld));

        residuals[0] = res[0]* res[0];
        return true;


    }

    const Vector3 gazeDirection;
    const Vector3 refDirection;
};



bool lineLineCalibration(const std::vector<Vector3>& refDirections, const std::vector<Vector3>& gazeDirections ,
    double (&orientation)[4], double (&translation)[3] , double& avgDistance, bool fixTranslation = false )
{

    // don't use Constructor 'Quaternion (const Scalar *data)' because the internal layout for coefficients is different from the one we use.
    // Memory Layout EIGEN: xyzw
    // Memory Layout CERES and the one we use: wxyz
    Eigen::Quaterniond q(orientation[0],orientation[1],orientation[2],orientation[3]); // don't map orientation
    Vector3 t =  Vector3(translation[0],translation[1], translation[2]);
    double n = t.norm();
    translation[0] /= n;
    translation[1] /= n;
    translation[2] /= n;

    Problem problem;
    double epsilon = std::numeric_limits<double>::epsilon();

    for(int i=0; i<refDirections.size(); i++) {

        auto gaze = gazeDirections.at(i);
        auto ref = refDirections.at(i);
        gaze.normalize(); //just to be sure
        ref.normalize(); //just to be sure

        // do a check to handle parameters we can't solve
        // First: the length of the directions must not be zero
        // Second: the angle between gaze direction and reference direction must not be greater 90 degrees, considering the initial orientation
        bool valid = true;
        valid |= gaze.norm() >= epsilon;
        valid |= ref.norm() >= epsilon;
        valid |= (q*gaze).dot(ref) >= epsilon;

        if( valid ){

            CostFunction* cost = new AutoDiffCostFunction<CoplanarityError , 1, 4, 3 >(new CoplanarityError(ref, gaze ));
            // TODO use a loss function, to handle gaze point outliers
            problem.AddResidualBlock(cost, nullptr, orientation,  translation );
        }else{
            std::cout << "no valid direction vector"  << std::endl;
        }
    }

    if( problem.NumResidualBlocks() == 0 ){
        std::cout << "nothing to solve"  << std::endl;
        return false;
    }

    ceres::LocalParameterization* quaternionParameterization = new ceres::QuaternionParameterization; // owned by the problem
    problem.SetParameterization(orientation, quaternionParameterization);

    ceres::LocalParameterization* normedTranslationParameterization = new pupillabs::Fixed3DNormParametrization(1.0); // owned by the problem
    problem.SetParameterization(translation, normedTranslationParameterization);

    if (fixTranslation)
    {
        problem.SetParameterBlockConstant(translation);
    }

    // Build and solve the problem.
    Solver::Options options;
    options.max_num_iterations = 1000;
    //options.linear_solver_type = ceres::DENSE_QR;

    options.parameter_tolerance = 1e-15;
    options.function_tolerance = 1e-16;
    options.gradient_tolerance = 1e-20;
   // options.minimizer_progress_to_stdout = true;
    //options.logging_type = ceres::SILENT;

    options.check_gradients = true;
    Solver::Summary summary;

    Solve(options, &problem, &summary);

    // std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    if( summary.termination_type != ceres::TerminationType::CONVERGENCE  ){
        std::cout << "Termination Error: " << ceres::TerminationTypeToString(summary.termination_type) << std::endl;
        return false;
    }

    //rescale the translation according to the initial translation
    translation[0] *= n;
    translation[1] *= n;
    translation[2] *= n;

    using singleeyefitter::Line3;
    // check for possible ambiguity
    //intersection points need to lie in positive z

    auto checkResult = [ &gazeDirections, &refDirections ]( Eigen::Quaterniond& orientation , Vector3 translation, double& avgDistance ){

        int validCount = 0;
        avgDistance = 0.0;
        for(int i=0; i<refDirections.size(); i++) {

            auto gaze = gazeDirections.at(i);
            auto ref = refDirections.at(i);

            gaze.normalize(); //just to be sure
            ref.normalize(); //just to be sure

            Vector3 gazeWorld = orientation * gaze;

            Line3 refLine = { Vector3(0,0,0) , ref  };
            Line3 gazeLine = { translation , gazeWorld  };

            auto closestPoints = closest_points_on_line( refLine , gazeLine );

            if( closestPoints.first.z() > 0.0 && closestPoints.second.z() > 0.0)
                validCount++;

            avgDistance += euclidean_distance( closestPoints.first, closestPoints.second );
        }
        avgDistance /= refDirections.size();
        return validCount == refDirections.size();
    };


    Eigen::Quaterniond q1(orientation[0],orientation[1],orientation[2],orientation[3]); // don't mapp orientation
    Vector3 t1 =  Vector3(translation[0],translation[1], translation[2]);
    Eigen::Quaterniond q2  = q1.conjugate();
    Vector3 t2 =  -t1;

    if(checkResult(q1,t1,avgDistance)){
        std::cout << "result one" <<std::endl;
        return true;
    }
    if(checkResult(q1,t2,avgDistance)){
        std::cout << "result two" <<std::endl;
        translation[0] *= -1.0;
        translation[1] *= -1.0;
        translation[2] *= -1.0;
        return true;
    }
    if(checkResult(q2,t1,avgDistance)){
        std::cout << "result three" <<std::endl;

        orientation[1] *= -1.0;
        orientation[2] *= -1.0;
        orientation[3] *= -1.0;
        return true;
    }
    if(checkResult(q2,t2,avgDistance)){
        std::cout << "result four" <<std::endl;

        orientation[1] *= -1.0;
        orientation[2] *= -1.0;
        orientation[3] *= -1.0;

        translation[0] *= -1.0;
        translation[1] *= -1.0;
        translation[2] *= -1.0;
        return true;
    }


    // we need to take the sphere position into account
    // thus the actual translation is not right, because the local coordinate frame of the eye need to be translated in the opposite direction
    // of the sphere coordinates

    // // since the actual translation is in world coordinates, the sphere translation needs to be calculated in world coordinates

    // Eigen::Matrix4d eyeToWorld =  Eigen::Matrix4d::Identity();
    // eyeToWorld.block<3,3>(0,0) = Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> >(rotation.data());
    // eyeToWorld(0, 3) = translation[0]*translationFactor;
    // eyeToWorld(1, 3) = translation[1]*translationFactor;
    // eyeToWorld(2, 3) = translation[2]*translationFactor;

    // Eigen::Vector4d sphereWorld = eyeToWorld * Eigen::Vector4d(spherePosition[0],spherePosition[1],spherePosition[2], 1.0 );
    // Vector3 sphereOffset =  sphereWorld.head<3>() - Vector3(translation);
    // Vector3 actualtranslation =  Vector3(translation) - sphereOffset;
    // // write the actual one back
    // translation[0] = actualtranslation[0];
    // translation[1] = actualtranslation[1];
    // translation[2] = actualtranslation[2];
    return true;

}

