

#ifndef singleeyefitter_CircleGoodness3D_h__
#define singleeyefitter_CircleGoodness3D_h__

#include "common/types.h"
#include "distance.h"
#include "common/constants.h"
#include "Geometry/Sphere.h"


namespace singleeyefitter {

    template<typename Scalar>
    class CircleGoodness3D {

        public:
            CircleGoodness3D()
            {
            };

            Scalar operator()(const Circle& circle, const Contours3D& contours) const
            {
                // how much does the contour support a circle ?
                Scalar points_length = 0.0;

                for (const auto& contour : contours) {
                    points_length = euclidean_distance(contour);
                }

                //std::cout << "Points length: " << points_length << std::endl;
                Scalar goodness =  points_length / (2.0 * constants::pi * circle.radius);

                if (goodness > 1.0) {
                    goodness = 1.0 - goodness;
                }

                return goodness;
            }
    };

} // namespace singleeyefitter

#endif // singleeyefitter_CircleGoodness3D_h__
