
/*
 Created by Lloyd Hughes on 2014/04/11.
 Copyright (c) 2014 Lloyd Hughes. All rights reserved.
 hughes.lloyd@gmail.com
*/


#ifndef EIGENQUATERNIONPARAMETERIZATION_H__
#define EIGENQUATERNIONPARAMETERIZATION_H__


#include <ceres/local_parameterization.h>
#include <Eigen/Core>


namespace pupillabs {

// Plus(x, delta) = [cos(|delta|), sin(|delta|) delta / |delta|] * x
// with * being the quaternion multiplication operator. Here we assume
// that the first element of the quaternion vector is the real (cos
// theta) part.
class EigenQuaternionParameterization : public ceres::LocalParameterization {
public:
    virtual ~EigenQuaternionParameterization() {}

    virtual bool Plus(const double* x_raw, const double* delta_raw, double* x_plus_delta_raw) const {
        const Eigen::Map<const Eigen::Quaterniond> x(x_raw);
        const Eigen::Map<const Eigen::Vector3d > delta(delta_raw);

        Eigen::Map<Eigen::Quaterniond> x_plus_delta(x_plus_delta_raw);

        const double delta_norm = delta.norm();
        if ( delta_norm > 0.0 ){
            const double sin_delta_by_delta = sin(delta_norm) / delta_norm;
            Eigen::Quaterniond tmp( cos(delta_norm), sin_delta_by_delta*delta[0], sin_delta_by_delta*delta[1], sin_delta_by_delta*delta[2] );

            x_plus_delta = tmp*x;
        }
        else {
            x_plus_delta = x;
        }
        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
            jacobian[0] =  x[3]; jacobian[1]  =  x[2]; jacobian[2]   = -x[1];  // NOLINT x
        jacobian[3] = -x[2]; jacobian[4]  =  x[3]; jacobian[5]   =  x[0];  // NOLINT y
        jacobian[6] =  x[1]; jacobian[7]  = -x[0]; jacobian[8]   =  x[3];  // NOLINT z
        jacobian[9] = -x[0]; jacobian[10] = -x[1]; jacobian[11] = -x[2];  // NOLINT w
            return true;
    }

    virtual int GlobalSize() const { return 4; }
    virtual int LocalSize() const { return 3; }

};


} // pupillabs


#endif /* end of include guard: EIGENQUATERNIONPARAMETERIZATION_H__ */
