

#ifndef singleeyefitter_CircleDeviationVariance3D_h__
#define singleeyefitter_CircleDeviationVariance3D_h__

#include "common/types.h"
#include "math/distance.h"
#include "common/constants.h"
#include "geometry/Sphere.h"


namespace singleeyefitter {

    template<typename Scalar>
    class CircleDeviationVariance3D {

        public:
            CircleDeviationVariance3D()
            {
            };

            Scalar operator()(const Circle& circle, const Contours3D& contours) const
            {


                Scalar residual_sqrt = 0.0;
                std::size_t size = 0;
                for (const auto& contour : contours) {
                    for (const auto& point : contour) {
                        residual_sqrt +=  std::abs(circle.radius * circle.radius  - euclidean_distance_squared(circle.center, point));
                    }
                    size += contour.size();
                }
                Scalar variance =  residual_sqrt / size;
                return variance;
            }
    };

} // namespace singleeyefitter

#endif // singleeyefitter_CircleDeviationVariance3D_h__
