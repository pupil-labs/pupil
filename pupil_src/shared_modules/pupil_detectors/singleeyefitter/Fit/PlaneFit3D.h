
#include <cstdio>
#include <vector>
#include <ceres/ceres.h>
#include <Eigen/Core>

#include "common/types.h"
#include "../mathHelper.h"


namespace singleeyefitter {

    template< typename Scalar >
    class PlaneFitter3D {
            typedef Eigen::Matrix<Scalar, 3, 1> Vector3;

        public:
            PlaneFitter3D() {};
            Vector3 getNormal() const {return mNormal; };
            Vector3 getPlanePoint() const {return mPoint; };

        private:
            Vector3 mNormal;
            Vector3 mPoint;

        public:

            bool fit(const std::vector<Vector3>& points )
            {

                int numPoints = points.size();

                if (numPoints < 3) {
                    return false;
                }

                // Compute the mean of the points. // could do this together with covariance
                Vector3 mean = Vector3(0, 0, 0);

                for (int i = 0; i < numPoints; ++i) {
                    mean += points[i];
                }

                mean /= numPoints;
                // Compute the covariance matrix of the points.
                double covar00 = 0.0, covar01 = 0.0, covar02 = 0.0;
                double covar11 = 0.0, covar12 = 0.0, covar22 = 0.0;

                for (int i = 0; i < numPoints; ++i) {
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

                mPoint = mean;

                // The plane normal is the eigenvector in the direction of smallest
                // variance of the points.
                mNormal =  eigen_solver.eigenvectors().col(0);
                mNormal.normalize();
                return true;

            }

            Scalar calculateResidual(const std::vector<Vector3>& points) const
            {
                Scalar error = 0.0;

                for (const auto& point : points) {
                    Vector3 diff = point - mPoint;
                    Scalar dot = diff.dot(mNormal);
                    error += std::abs( dot * dot);
                }

                error /= points.size();
                return error;

            }

            Scalar calculateResidual(const Vector3& point) const
            {
                Vector3 diff = point - mPoint;
                Scalar dot = diff.dot(mNormal);
                Scalar error = std::abs( dot * dot);
                return error;
            }
    };

} // namespace singleeyefitter
