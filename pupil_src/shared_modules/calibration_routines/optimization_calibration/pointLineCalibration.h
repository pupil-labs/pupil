


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

struct TransformationError {
    TransformationError(const Vector3 referencePoint,   const Vector3 linePoint1, const Vector3 linePoint2)
        : referencePoint(referencePoint), linePoint1(linePoint1), linePoint2(linePoint2) {}

    template <typename T>
    bool operator()(
        const T* const orientation,  // orientation denoted by quaternion
        const T* const translation,  // followed by translation
        T* residuals) const
    {


        // Compute coordinates with current transformation matrix: y = Rx + t.

        Eigen::Matrix<T, 3, 1> l1 = {T(linePoint1[0]), T(linePoint1[1]), T(linePoint1[2])};
        Eigen::Matrix<T, 3, 1> l2 = {T(linePoint2[0]), T(linePoint2[1]), T(linePoint2[2])};
        Eigen::Matrix<T, 3, 1> refP = {T(referencePoint[0]) , T(referencePoint[1]) , T(referencePoint[2])};
        Eigen::Matrix<T, 3, 1> t = {T(translation[0]) , T(translation[1]) , T(translation[2])};

        Eigen::Matrix<T, 3, 1> p1;
        Eigen::Matrix<T, 3, 1> p2;

        //rotate
        ceres::QuaternionRotatePoint(orientation, l1.data(), p1.data() );
        ceres::QuaternionRotatePoint(orientation, l2.data(), p2.data() );
        //translate
        p1 += t;
        p2 += t;

        // Equation 3: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        // delta tells us if the point lies "behind" or "infront" of p1
        T delta = -(p1 - refP ).dot(p2 - p1) / (p2 - p1).squaredNorm();

        // in our case the point should alway lay on the ray from p1 to p2
        if(  delta  >= 0.0 ){
            // now calculate the distance between the observed point and the nearest point on the line
            // Equation 10: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
            // and divide by delta
            // by dividing by delta we actually optimize the Sine of the angle between these two lines
            residuals[0] = ((refP - p1).cross(refP - p2).squaredNorm() / (p2 - p1).squaredNorm()) / (delta*delta);
            return true;

        }
        return false;


    }

    const Vector3 referencePoint;
    const Vector3 linePoint1;
    const Vector3 linePoint2;
};


void pointLineCalibration(Vector3 spherePosition, const std::vector<Vector3>& refPoints, const std::vector<Vector3>& gazeDirections , double* orientation , double* translation )
{

    // don't use Constructor 'Quaternion (const Scalar *data)' because the internal layout for coefficients is different from the one we use.
    // Memory Layout EIGEN: xyzw
    // Memory Layout CERES and the one we use: wxyz
    Eigen::Quaterniond q(orientation[0],orientation[1],orientation[2],orientation[3]);

    Problem problem;
    int i = 0;

    for (auto& p : refPoints) {
        auto g = gazeDirections.at(i);
        i++;
        g.normalize();

        // do a check to handle parameters we can't solve
        // First: the length of the line must be greater zero
        // Second: the angle between line direction and reference point direction must not be greater 90 degrees, considering the initial orientation
        auto v = q*g;
        if( g.norm() >= std::numeric_limits<double>::epsilon() && v.dot(p) > 0.0   ){
            CostFunction* cost = new AutoDiffCostFunction<TransformationError , 1, 4, 3 >(new TransformationError(p , Vector3::Zero() , g ));
            problem.AddResidualBlock(cost, nullptr, orientation,  translation);
        }else{
            std::cout << "no valid direction vector"  << std::endl;
        }
    }

    if( problem.NumResidualBlocks() == 0 ){
        std::cout << "nothing to solve"  << std::endl;
        return;
    }

    problem.SetParameterBlockConstant(translation);

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


    //Ceres Matrices are RowMajor, where as Eigen is default ColumnMajor
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotation;
    ceres::QuaternionToRotation( orientation , rotation.data() );
    // ceres should always return a valid quaternion
    // double det = r.determinant();
    // std::cout << "det:: " << det << std::endl;
    // if(  det == 1 ){
    //     std::cout << "Error: No valid rotation matrix."   << std::endl;
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


}

