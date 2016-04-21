/*

Copyright (C) 2014, University of Oulu, all rights reserved.
Copyright (C) 2014, NVIDIA Corporation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the name of UNIVERSITY OF OULU, NVIDIA CORPORATION nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef FIXED3DNORMPARAMETRIZATION_H__
#define FIXED3DNORMPARAMETRIZATION_H__


#include <ceres/local_parameterization.h>
#include <Eigen/Core>

namespace pupillabs
{

/**
 * @brief A parameterization class that is used for CERES solver. It parametrizes a 3D vector (like a translation) with two components, keeping the L2 norm fixed
 */
class Fixed3DNormParametrization: public ceres::LocalParameterization
{
public:
    Fixed3DNormParametrization(double norm)
            : mFixedNorm(norm)
    {
    }
    virtual ~Fixed3DNormParametrization()
    {
    }

    virtual int GlobalSize() const
    {
        return 3;
    }
    virtual int LocalSize() const
    {
        return 2;
    }

    // Calculates two vectors that are orthogonal to X.
    // It first picks a non-colinear point C then basis1=(X-C) x C and basis2=X x basis1
    static void GetBasis(const double *x, double *basis1, double *basis2)
    {
        const double kThreshold = 0.1;

        //Check that the point we use is not colinear with x
        if (x[1] > kThreshold || x[1] < -kThreshold || x[2] > kThreshold || x[2] < -kThreshold)
        {
            //Use C=[1,0,0]
            basis1[0] = 0;
            basis1[1] = x[2];
            basis1[2] = -x[1];

            basis2[0] = -(x[1] * x[1] + x[2] * x[2]);
            basis2[1] = x[0] * x[1];
            basis2[2] = x[0] * x[2];
        }
        else
        {
            //Use C=[0,1,0]
            basis1[0] = -x[2];
            basis1[1] = 0;
            basis1[2] = x[0];

            basis2[0] = x[0] * x[1];
            basis2[1] = -(x[0] * x[0] + x[2] * x[2]);
            basis2[2] = x[1] * x[2];
        }
        double norm;
        norm = sqrt(basis1[0] * basis1[0] + basis1[1] * basis1[1] + basis1[2] * basis1[2]);
        basis1[0] /= norm;
        basis1[1] /= norm;
        basis1[2] /= norm;

        norm = sqrt(basis2[0] * basis2[0] + basis2[1] * basis2[1] + basis2[2] * basis2[2]);
        basis2[0] /= norm;
        basis2[1] /= norm;
        basis2[2] /= norm;

    }

    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        double basis1[3];
        double basis2[3];

        //Translation is constrained
        GetBasis(x, basis1, basis2);

        x_plus_delta[0] = x[0] + delta[0] * basis1[0] + delta[1] * basis2[0];
        x_plus_delta[1] = x[1] + delta[0] * basis1[1] + delta[1] * basis2[1];
        x_plus_delta[2] = x[2] + delta[0] * basis1[2] + delta[1] * basis2[2];

        double norm = sqrt(
                x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] + x_plus_delta[2] * x_plus_delta[2]);
        double factor = mFixedNorm / norm;
        x_plus_delta[0] *= factor;
        x_plus_delta[1] *= factor;
        x_plus_delta[2] *= factor;

        return true;
    }

    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
        typedef Eigen::Matrix<double, 3,2> Matrix32d;
        Matrix32d &jacobian_ = *(Matrix32d *)jacobian;
        double basis1[3];
        double basis2[3];

        //Translation is special
        GetBasis(x, basis1, basis2);

        jacobian_(0, 0) = basis1[0];
        jacobian_(1, 0) = basis1[1];
        jacobian_(2, 0) = basis1[2];

        jacobian_(0, 1) = basis2[0];
        jacobian_(1, 1) = basis2[1];
        jacobian_(2, 1) = basis2[2];
        return true;
    }


protected:
    const double mFixedNorm;
};

} //namespace pupillabs


#endif /* end of include guard: FIXED3DNORMPARAMETRIZATION_H__ */
