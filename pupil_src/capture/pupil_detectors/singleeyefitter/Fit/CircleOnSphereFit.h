
#ifndef CIRCLEFIT_H__
#define CIRCLEFIT_H__


#include "common/types.h"
#include "geometry/Sphere.h"
#include "PlaneFit3D.h"


namespace singleeyefitter {

    template<typename Scalar>
    class CircleOnSphereFitter {

    public:
            CircleOnSphereFitter( const Sphere<Scalar> sphere ) : mSphere(sphere)
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

                // check if the normal points outwards or inwards
                // the plane normal must lie in the same direction as the planepoint from sphere center
                if( (planePoint - mSphere.center).dot(normal) < 0 )
                    normal *= -1.0;

                // calculate circle radius and circle center

                //first calculate the distance from sphere center to plane
                double d = std::abs((mSphere.center - planePoint ).dot(normal) /  normal.norm() );
                // than calculate intersection of normal and plane
                Vector3 circle_center = mSphere.center + d * normal;
                // calculate circle radius  r = sqrt(R^2 - d^2) // R is sphere radius
                double r = sqrt( mSphere.radius*mSphere.radius - d * d);


                mCircle = {circle_center, normal, r};
                return true;

            }

            Scalar calculateResidual(const std::vector<Vector3>& points) const
            {
                return  mPlaneFitter.calculateResidual(points);
            }
            const Circle& getCircle() { return mCircle;};


        private:
            Circle mCircle;
            Sphere<Scalar> mSphere;
            PlaneFitter3D<Scalar> mPlaneFitter;



    };


} // singleeyefitter



#endif /* end of include guard: CIRCLEFIT_H__ */
