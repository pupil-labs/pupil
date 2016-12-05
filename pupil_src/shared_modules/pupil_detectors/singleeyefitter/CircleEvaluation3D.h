

#ifndef singleeyefitter_CircleEvaluation3D_h__
#define singleeyefitter_CircleEvaluation3D_h__

#include "common/types.h"
#include "math/distance.h"
#include "common/constants.h"
#include "geometry/Sphere.h"


namespace singleeyefitter {

    template<typename Scalar>
    class CircleEvaluation3D {

        public:
            CircleEvaluation3D(const Vector3 camera_center, const Sphere<Scalar> sphere, const Scalar max_residual, const Scalar circle_radius_min, const Scalar circle_radius_max) :
                camera_center(camera_center), sphere(sphere),  max_residual(max_residual), circle_radius_min(circle_radius_min), circle_radius_max(circle_radius_max)
            {
            };

            bool operator()(const Circle& circle , Scalar residual) const
            {

                // Circle normal must lay in camera direction +- 90degree sphere center and camera line
                Scalar normalDotPos = circle.normal.dot(camera_center - sphere.center);

                // it has to face the camera
                if (normalDotPos <= 0) {
                    //std::cout << "not facing camera" << std::endl;
                    return false;
                }

                // reject if radius is too small or big
                if (circle.radius < circle_radius_min || circle.radius > circle_radius_max) {
                    //std::cout << "radius not in range" << std::endl;
                    return false;
                }

                // also the residual must not be too high
                if (residual > max_residual) {
                    return false;
                }

                // else this circle could be a candidate
                return true;

            }

        private:
            Vector3 camera_center;
            Sphere<Scalar> sphere;
            float max_residual;
            float circle_radius_min, circle_radius_max;

    };

} // namespace singleeyefitter

#endif // singleeyefitter_CircleEvaluation3D_h__
