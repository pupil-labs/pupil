





#include <cstdio>
#include <vector>
#include <ceres/ceres.h>
#include "../common/types.h"
#include "../mathHelper.h"





singleeyefitter::Vector3 fit_plan( std::vector<singleeyefitter::Vector3>& points ){

    using singleeyefitter::Vector3;
    int numPoints = points.size();
    if ( numPoints < 3){
        std::cout << "point size must be at least 3" << std::endl;
        return singleeyefitter::Vector3();
    }

    // Compute the mean of the points. // could do this together with covariance
    Vector3 mean = Vector3(0,0,0);
    for (int i = 0; i < numPoints; ++i)
    {
        mean += points[i];
    }
    mean /= numPoints;

    // Compute the covariance matrix of the points.
    double covar00 = 0.0, covar01 = 0.0, covar02 = 0.0;
    double covar11 = 0.0, covar12 = 0.0, covar22 = 0.0;
    for (int i = 0; i < numPoints; ++i)
    {
        Vector3 diff = points[i] - mean;
        covar00 += diff[0] * diff[0];
        covar01 += diff[0] * diff[1];
        covar02 += diff[0] * diff[2];
        covar11 += diff[1] * diff[1];
        covar12 += diff[1] * diff[2];
        covar22 += diff[2] * diff[2];
    }

    Eigen::Matrix3d adjoint_matrix;

    adjoint_matrix <<   covar00 , covar01, covar02,
                        covar01 , covar11, covar12,
                        covar02 , covar12, covar22;
    // Solve the eigensystem.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(adjoint_matrix);


    if(eigen_solver.eigenvalues()[0] > eigen_solver.eigenvalues()[1])
        std::cout << "not unique" << std::endl;

    // The plane normal is the eigenvector in the direction of smallest
    // variance of the points.
    return eigen_solver.eigenvectors().col(0);


}


// template <typename Real>
// Real ApprOrthogonalPlane3<Real>::Error(Vector3<Real> const& observation)
// const
// {
//     Vector3<Real> diff = observation - mParameters.first;
//     Real sqrlen = Dot(diff, diff);
//     Real dot = Dot(diff, mParameters.second);
//     Real error = std::abs(sqrlen - dot*dot);
//     return error;
// }
