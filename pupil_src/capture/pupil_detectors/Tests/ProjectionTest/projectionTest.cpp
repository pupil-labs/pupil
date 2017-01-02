/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/


#include <iostream>
#include "../../singleeyefitter/projection.h"
#include "geometry/Ellipse.h"
#include "geometry/Conic.h"
#include "common/types.h"



int main()
{

    using namespace singleeyefitter;

    std::cout << "Start Test" << std::endl;
    const Ellipse ellipse(30.0, 1.0, 1, 0.9, 0);
    double circle_radius = 1;
    double focal_length  = 50;

    auto pair = unproject(ellipse, circle_radius, focal_length);
    std::cout << pair.first  << std::endl;
    std::cout << pair.second  << std::endl;

    auto conic_unproj_first = project( pair.first, focal_length);
    auto conic_unproj_second = project( pair.second, focal_length);

    std::cout <<  Ellipse(conic_unproj_first)  << std::endl;
    std::cout <<  Ellipse(conic_unproj_second)  << std::endl;

    //auto conic_ellipse = Conic<double>(ellipse);
   // std::cout <<  Ellipse(conic_ellipse)  << std::endl;



}
