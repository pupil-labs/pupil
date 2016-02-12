


#include "common.h"
#include <vector>
#include <cstdio>
#include <limits>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Geometry>


using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

template<typename Scalar>
struct Result{
    Scalar distanceSquared;
    bool valid = false;
};

// since for rayray distance both parameters s,t for eq r0=p0+s*d0 and r1=p1+t*d1 need to be positive
// ceres get's to know if the rays don't lie in the same direction with angle less than 90 degree
// Book: Geometric Tools for Computer Graphics, Side 413
template<typename Scalar, int Dim>
Result<Scalar> ceresRayRayDistanceSquared(const Eigen::ParametrizedLine<Scalar, Dim>& ray0, const Eigen::ParametrizedLine<Scalar, Dim>& ray1 )
{

    typedef typename Eigen::ParametrizedLine<Scalar, Dim>::VectorType Vector;

    Result<Scalar> result;
    result.valid = false;

    Vector diff = ray0.origin() - ray1.origin();
    Scalar a01 = - ray0.direction().dot(ray1.direction());
    Scalar b0 = diff.dot(ray0.direction());
    Scalar b1;
    Scalar s0, s1;

    if (ceres::abs(a01) < Scalar(1) )
    {
        // Rays are not parallel.
        b1 = -diff.dot(ray1.direction());
        s0 = a01 * b1 - b0;
        s1 = a01 * b0 - b1;

        if (s0 >= Scalar(0) )
        {
            if (s1 >= Scalar(0) )
            {
                // Minimum at two  points of rays.
                Scalar det = Scalar(1) - a01 * a01;
                s0 /= det;
                s1 /= det;

                Vector closestPoint0 = ray0.origin() + s0 * ray0.direction();
                Vector closestPoint1 = ray1.origin() + s1 * ray1.direction();
                diff = closestPoint0 - closestPoint1;
                result.distanceSquared =  diff.dot(diff);
                result.valid = true;
                return result;
            }
        }
    }
    // everything else is not valid
    return result;

}



struct TransformationRayRayError {
    TransformationRayRayError(const Vector3 refDirection,   const Vector3 gazeDirection )
        : refDirection(refDirection), gazeDirection(gazeDirection) {}

    template <typename T>
    bool operator()(
        const T* const orientation,  // orientation denoted by quaternion
        const T* const translation,  // followed by translation
        T* residuals) const
    {

        // Compute coordinates with current transformation matrix: y = Rx + t.
        Eigen::Matrix<T, 3, 1> gazeP = {T(gazeDirection[0]), T(gazeDirection[1]), T(gazeDirection[2])};
        Eigen::Matrix<T, 3, 1> refP = {T(refDirection[0]) , T(refDirection[1]) , T(refDirection[2])};
        Eigen::Matrix<T, 3, 1> t = {T(translation[0]) , T(translation[1]) , T(translation[2])};

        Eigen::Matrix<T, 3, 1> gazeTransformed;

        //rotate
        ceres::QuaternionRotatePoint(orientation, gazeP.data(), gazeTransformed.data() );
        //ceres::QuaternionRotatePoint(orientation, l2.data(), p2.data() );

        //translate

        Eigen::Matrix<T, 3, 1> origin = {T(0),T(0),T(0)};
        Eigen::ParametrizedLine<T, 3> gazeLine = {t , gazeTransformed};
        Eigen::ParametrizedLine<T, 3> refLine = {origin, refP };

        Result<T> result = ceresRayRayDistanceSquared(gazeLine , refLine);

        if(  result.valid ){
            residuals[0] = result.distanceSquared;
            return true;
        }
        return false;

    }

    const Vector3 gazeDirection;
    const Vector3 refDirection;
};



bool lineLineCalibration(Vector3 spherePosition, const std::vector<Vector3>& refDirections, const std::vector<Vector3>& gazeDirections ,
    double* orientation , double* translation , bool fixTranslation = false ,
    Vector3 translationLowerBound = {15,5,5},Vector3 translationUpperBound = {15,5,5}
    )
{

    // don't use Constructor 'Quaternion (const Scalar *data)' because the internal layout for coefficients is different from the one we use.
    // Memory Layout EIGEN: xyzw
    // Memory Layout CERES and the one we use: wxyz
    Eigen::Quaterniond q(orientation[0],orientation[1],orientation[2],orientation[3]);

    Problem problem;
    double epsilon = std::numeric_limits<double>::epsilon();

    for(int i=0; i<refDirections.size(); i++) {

        auto gaze = gazeDirections.at(i);
        auto ref = refDirections.at(i);
        gaze.normalize(); //just to be sure
        ref.normalize(); //just to be sure
        i++;

        // do a check to handle parameters we can't solve
        // First: the length of the directions must not be zero
        // Second: the angle between gaze direction and reference direction must not be greater 90 degrees, considering the initial orientation

        bool valid = true;
        valid |= gaze.norm() >= epsilon;
        valid |= ref.norm() >= epsilon;
        valid |= (q*gaze).dot(ref) >= epsilon;

        if( valid ){
            CostFunction* cost = new AutoDiffCostFunction<TransformationRayRayError , 1, 4, 3 >(new TransformationRayRayError(ref, gaze ));
            // TODO use a loss function, to handle gaze point outliers
            problem.AddResidualBlock(cost, nullptr, orientation,  translation);
        }else{
            std::cout << "no valid direction vector"  << std::endl;
        }
    }

    if( problem.NumResidualBlocks() == 0 ){
        std::cout << "nothing to solve"  << std::endl;
        return false;
    }

    if (fixTranslation)
    {
        problem.SetParameterBlockConstant(translation);
    }else{

        Vector3 upperBound = Vector3(translation) + translationUpperBound;
        Vector3 lowerBound = Vector3(translation) - translationLowerBound;

        problem.SetParameterLowerBound(translation, 0 , lowerBound[0] );
        problem.SetParameterLowerBound(translation, 1 , lowerBound[1] );
        problem.SetParameterLowerBound(translation, 2 , lowerBound[2] );

        problem.SetParameterUpperBound(translation, 0 , upperBound[0] );
        problem.SetParameterUpperBound(translation, 1 , upperBound[1] );
        problem.SetParameterUpperBound(translation, 2 , upperBound[2] );
    }



    ceres::LocalParameterization* quaternionParameterization = new ceres::QuaternionParameterization; // owned by the problem
    problem.SetParameterization(orientation, quaternionParameterization);

    // Build and solve the problem.
    Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_QR;
    //options.parameter_tolerance = 1e-14;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-20;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    //options.check_gradients = true;
    Solver::Summary summary;

    Solve(options, &problem, &summary);

    // std::cout << summary.BriefReport() << "\n";
    //std::cout << summary.FullReport() << "\n";

    if( summary.termination_type != ceres::TerminationType::CONVERGENCE  ){
        std::cout << "Termination Error: " << ceres::TerminationTypeToString(summary.termination_type) << std::endl;
        return false;
    }

    //Ceres Matrices are RowMajor, where as Eigen is default ColumnMajor
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotation;
    ceres::QuaternionToRotation( orientation , rotation.data() );
    // ceres should always return a valid quaternion
    // double det = r.determinant();
    // std::cout << "det:: " << det << std::endl;
    // if(  det == 1 ){
    //     std::cout << "Error: No valid rotation matrix."   << std::endl;
    //     return false;
    // }


    // we need to take the sphere position into account
    // thus the actual translation is not right, because the local coordinate frame of the eye need to be translated in the opposite direction
    // of the sphere coordinates

    // since the actual translation is in world coordinates, the sphere translation needs to be calculated in world coordinates
    Eigen::Matrix4d eyeToWorld =  Eigen::Matrix4d::Identity();
    eyeToWorld.block<3,3>(0,0) = Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> >(rotation.data());
    eyeToWorld(0, 3) = translation[0];
    eyeToWorld(1, 3) = translation[1];
    eyeToWorld(2, 3) = translation[2];

    Eigen::Vector4d sphereWorld = eyeToWorld * Eigen::Vector4d(spherePosition[0],spherePosition[1],spherePosition[2], 1.0 );
    Vector3 sphereOffset =  sphereWorld.head<3>() - Vector3(translation);
    Vector3 actualtranslation =  Vector3(translation) - sphereOffset;
    // write the actual one back
    translation[0] = actualtranslation[0];
    translation[1] = actualtranslation[1];
    translation[2] = actualtranslation[2];
    return true;

}

