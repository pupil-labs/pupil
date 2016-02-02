


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
    TransformationError(const Vector3 referencePoint,   const Vector3 gazePoint1, const Vector3 gazePoint2)
        : referencePoint(referencePoint), gazePoint1(gazePoint1), gazePoint2(gazePoint2) {}

    template <typename T>
    bool operator()(
        const T* const R,  // Rotation denoted by quaternion
        const T* const t,
        // followed with translation
        T* residuals) const
    {


        // Compute coordinates with current transformation matrix: x = RX + t.


        T g1[3] = {T(gazePoint1[0]), T(gazePoint1[1]), T(gazePoint1[2])};
        T g2[3] = {T(gazePoint2[0]), T(gazePoint2[1]), T(gazePoint2[2])};

        T p1[3];
        T p2[3];
        ceres::QuaternionRotatePoint(R, g1, p1 );
        ceres::QuaternionRotatePoint(R, g2, p2 );

        p1[0] += t[0];
        p1[1] += t[1];
        p1[2] += t[2];
        p2[0] += t[0];
        p2[1] += t[1];
        p2[2] += t[2];

        Eigen::Matrix<T, 3, 1> refP;
        refP << T(referencePoint[0]) , T(referencePoint[1]) , T(referencePoint[2]);

        Eigen::Matrix<T, 3, 1> ep1;
        ep1 << T(p1[0]) , T(p1[1]) , T(p1[2]);

        Eigen::Matrix<T, 3, 1> ep2;
        ep2 << T(p2[0]) , T(p2[1]) , T(p2[2]);
        // now calculate the distance between the observed point and the nearest point on the line
        residuals[0] = (refP - ep1).cross(refP - ep2).norm() / (ep2 - ep1).norm();
        return true;
    }

    const Vector3 referencePoint;
    const Vector3 gazePoint1;
    const Vector3 gazePoint2;
};


Eigen::Matrix4d pointLineCalibration(Vector3 spherePosition, const std::vector<Vector3>& refPoints, const std::vector<Vector3>& gazeDirections,  std::vector<Vector3>& gazePoints)
{



    Problem problem;

    double angle_axis[3] = {0.0,0,0.0};
    double rotation[4];
    ceres::AngleAxisToQuaternion(  angle_axis ,rotation );
    double translation[3] = { 0.0 , 0.0  , 0.0 };


    ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;

    int i = 0;

    for (auto& p : refPoints) {
        auto g = gazeDirections.at(i);
        i++;
        CostFunction* cost = new AutoDiffCostFunction<TransformationError , 1, 4, 3 >(new TransformationError(p , g , spherePosition));
        problem.AddResidualBlock(cost, nullptr, &rotation[0],  &translation[0]);
    }

    problem.SetParameterBlockConstant(&translation[0]);

    problem.SetParameterization(&rotation[0], quaternion_parameterization);


    // Build and solve the problem.
    Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::DENSE_QR;
    options.parameter_tolerance = 1e-10;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    // options.minimizer_type = ceres::TRUST_REGION;
    //options.minimizer_progress_to_stdout = true;
    options.check_gradients = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    // // Recover r from m.

     std::cout << summary.BriefReport() << "\n";
    //std::cout << summary.FullReport() << "\n";
   for (int i = 0; i < 4; i++) {
        std::cout << "," << rotation[i] << std::endl;
    }
    for (int i = 0; i < 3; i++) {
        std::cout << "," << translation[i] << std::endl;
    }


    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> r;
    Eigen::Matrix<double, 3, 3> rc;
    ceres::QuaternionToRotation( rotation , r.data() );
    std::cout << "r: " << r << std::endl;
    std::cout << "det: " << r.determinant() << std::endl;
    rc = Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> >(r.data());




    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t.block<3,3>(0,0) = rc;
    t(0, 3) = translation[0];
    t(1, 3) = translation[1];
    t(2, 3) = translation[2];
    std::cout << "transformation: "  << t  << std::endl;

    // Eigen::Matrix4d tt = Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> >(t.data());
    // std::cout << "transformation: "  << tt  << std::endl;

    return t;
}

