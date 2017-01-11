/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/



#include "../singleeyefitter/utils.h" // random
#include "common/types.h"


std::vector<singleeyefitter::Vector3> createCirclePointsOnSphere( singleeyefitter::Vector2 center, double opening_angle_alpha,  int amount, float circle_segment_range, double randomAmount = 0.0 ){

    using namespace singleeyefitter;
    using std::sin;
    using std::cos;

    // http://math.stackexchange.com/questions/643130/circle-on-sphere
    double alpha = opening_angle_alpha;


    double sin_a = sin(alpha);
    double cos_a = cos(alpha);


    std::vector<Vector3> points;

    for (int i = 0; i < amount; ++i)
    {
        double beta = center[0] + M_PI * singleeyefitter::random(-randomAmount/2.0,randomAmount/2.0 );
        double gamma = center[1]+ M_PI/2 * singleeyefitter::random(-randomAmount/2.0,randomAmount/2.0 );
        double sin_b = sin(beta);
        double cos_b = cos(beta);
        double sin_g = sin(gamma);
        double cos_g = cos(gamma);

        double t = circle_segment_range *  2.0 * M_PI * float(i)/(amount-1);
        double cos_t = cos(t);
        double sin_t = sin(t);

        double z = sin_a * cos_b * cos_g * cos_t - sin_a * sin_g * sin_t + cos_a * sin_b * cos_g;
        double x =  sin_a * cos_b * sin_g * cos_t + sin_a * cos_g * sin_t + cos_a * sin_b * sin_g;
        double y = -sin_a * sin_b * cos_t + cos_a * cos_b;
        points.emplace_back(x,y,z);
    }

    return points;

}
