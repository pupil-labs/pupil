


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
        const T* const R_t,  // Rotation denoted by angle axis
        // followed with translation
        T* residuals) const
    {


        Eigen::Matrix<T, 3, 1> translation;
        translation[0] = R_t[3];
        translation[1] = R_t[4];
        translation[2] = R_t[5];
        // Compute coordinates with current transformation matrix: x = RX + t.
        Eigen::Matrix<T, 3, 1> p1;
        Eigen::Matrix<T, 3, 1> p2;

        T g1[3] = {T(gazePoint1[0]), T(gazePoint1[1]), T(gazePoint1[2])};
        T g2[3] = {T(gazePoint2[0]), T(gazePoint2[1]), T(gazePoint2[2])};

        ceres::AngleAxisRotatePoint(R_t, g1 , p1.data());
        ceres::AngleAxisRotatePoint(R_t, g2, p2.data());

        p1 += translation;
        p2 += translation;

        Eigen::Matrix<T, 3, 1> refP;
        refP << T(referencePoint[0]) , T(referencePoint[1]) , T(referencePoint[2]);

        // now calculate the distance between the observed point and the nearest point on the line
        residuals[0] = (refP - p1).cross(refP - p2).norm() / (p2 - p1).norm();
        return true;
    }

    const Vector3 referencePoint;
    const Vector3 gazePoint1;
    const Vector3 gazePoint2;
};


Eigen::Matrix4d pointLineCalibration(Vector3 spherePosition, const std::vector<Vector3>& refPoints, const std::vector<Vector3>& gazeDirections,  std::vector<Vector3>& gazePoints)
{



    Problem problem;
    double transformation[6] = {0.0 , -1.5 , 0.0 , -30.0 , -30.0  , -20.0 };


    // Parameterization used to restrict camera motion for modal solvers.
    ceres::SubsetParameterization* constant_transform_parameterization = NULL;
    std::vector<int> constant_translation;
    // First three elements are rotation, last three are translation.
    constant_translation.push_back(3);
    constant_translation.push_back(4);
    constant_translation.push_back(5);
    constant_transform_parameterization =
        new ceres::SubsetParameterization(6, constant_translation);


    int i = 0;

    for (auto& p : refPoints) {
        auto g = gazeDirections.at(i);
        i++;
        CostFunction* cost = new AutoDiffCostFunction<TransformationError , 1, 6 >(new TransformationError(p , g , spherePosition));
        problem.AddResidualBlock(cost, nullptr, &transformation[0]);
    }

    problem.SetParameterization(&transformation[0],
                                constant_transform_parameterization);


    // Build and solve the problem.
    Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::DENSE_QR;
    options.parameter_tolerance = 1e-10;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    // options.minimizer_type = ceres::TRUST_REGION;
    options.minimizer_progress_to_stdout = true;
    options.check_gradients = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    // // Recover r from m.

    // std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    for (int i = 0; i < 6; i++) {
        std::cout << "," << transformation[i] << std::endl;
    }

    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(transformation[0], Vector3::UnitX())
        * Eigen::AngleAxisd(transformation[1], Vector3::UnitY())
        * Eigen::AngleAxisd(transformation[2], Vector3::UnitZ());

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t.block<3, 3>(0, 0) = m;
    t(0, 3) = transformation[3];
    t(1, 3) = transformation[4];
    t(2, 3) = transformation[5];

    std::cout << "transformation: "  << t  << std::endl;
    return t;
}

