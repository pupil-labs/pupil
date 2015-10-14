

#include "../common/types.h"
#include "CircleOnSphereUtils.h"
#include <cmath>
#include <vector>
#include <iostream>


using namespace singleeyefitter;




int main(int argc, char** argv){


    // point lying on the unit sphere 0 < coord < PI
    Vector2 center(M_PI_2, M_PI_2 );
    auto points = createCirclePointsOnSphere( center , M_PI_4, 20 );
    for( auto p : points ){

        std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;

    }
    return 0;
}
