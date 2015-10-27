

#ifndef singleeyefitter_CircleGoodness3D_h__
#define singleeyefitter_CircleGoodness3D_h__

#include "common/types.h"
#include "distance.h"
#include "common/constants.h"
#include "Geometry/Sphere.h"


namespace singleeyefitter {

    template<typename Scalar>
    class CircleDeviationVariance3D {

        public:
            CircleDeviationVariance3D(const Vector3 camera_center, const Sphere<Scalar> sphere, const Scalar max_residual, const Scalar circle_radius_min, const Scalar circle_radius_max) :
                camera_center(camera_center), sphere(sphere),  max_residual(max_residual), circle_radius_min(circle_radius_min), circle_radius_max(circle_radius_max)
            {
            };

            Scalar operator()(const Circle& circle, const std::vector<Vector3>& points ) const
            {

                // Circle normal must lay in camera direction +- 90degree sphere center and camera line
                Scalar normalDotPos = circle.normal.dot(camera_center - sphere.center  );
                // the goodness actually doesn't really depend on the normal, but it has to face at least the camera
                if (normalDotPos <= 0 ) {
                    //std::cout << "not facing camera" << std::endl;
                    return std::numeric_limits<Scalar>::infinity();
                }
                // reject if radius is to small or big
                if( circle.radius < circle_radius_min || circle.radius > circle_radius_max){
                    //std::cout << "radius not in range" << std::endl;
                    return std::numeric_limits<Scalar>::infinity();;
                }

                // Scalar residual_sqrt = 0.0;
                // for( const auto& point : points){
                //    residual_sqrt +=  std::abs(circle.radius * circle.radius  - euclidean_distance_squared(circle.center, point));
                // }
                // Scalar variance =  residual_sqrt / points.size();


                Scalar points_length = 0.0;
                points_length = euclidean_distance(points);
                //std::cout << "Points length: " << points_length << std::endl;
                Scalar variance = std::abs( 1.0 - points_length/ (2.0 * constants::pi * circle.radius)  );

               return variance;
            }

        private:
            Vector3 camera_center;
            Sphere<Scalar> sphere;
            float max_residual;
            float circle_radius_min, circle_radius_max;

    };

} // namespace singleeyefitter

#endif // singleeyefitter_CircleGoodness3D_h__
