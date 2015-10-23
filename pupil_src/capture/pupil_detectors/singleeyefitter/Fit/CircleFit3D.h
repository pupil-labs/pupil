
#ifndef CIRCLEFIT_H__
#define CIRCLEFIT_H__


#include "../common/types.h"
#include "PlaneFit3D.h"


namespace singleeyefitter {

    template<typename Scalar>
    class CircleFitter3D {


            CircleFitter3D()
            {
                mPlaneFitter = PlaneFitter3D<Scalar>();
            }

            bool fit(const std::vector<Vector3>& points)
            {

                if (!mPlaneFitter.fit(points)) {
                    return false;
                }

                Vector3 normal = mPlaneFitter.getNormal();
                Vector3 planePoint = mPlaneFitter.getPlanePoint();

                // calculate circle radius
                //first calculate distance from sphere center to plane
                double d = planePoint.dot(normal) / normal.norm();
                // than calculate intersection of normal and plane
                Vector3 circle_center = d * normal;
                // calculate circle radius  r = sqrt(R^2 - d^2) // R is sphere radius
                double r = sqrt(1 - d * d);
                mCircle = {circle_center, normal, r};

            }

            Scalar calculateResidual(const std::vector<Vector3>& points) const
            {
                return  mPlaneFitter.calculateResidual(points);
            }
            Circle getCircle() { return mCircle;};


        private:
            Circle mCircle;
            PlaneFitter3D<Scalar> mPlaneFitter;



    };


} // singleeyefitter



#endif /* end of include guard: CIRCLEFIT_H__ */
