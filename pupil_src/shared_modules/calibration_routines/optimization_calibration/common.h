/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/

#ifndef COMMON_H__
#define COMMON_H__


#include <vector>


typedef Eigen::Matrix<double, 2, 1> Vector2;
typedef Eigen::Matrix<double, 3, 1> Vector3;
typedef Eigen::Matrix<double, 4, 1> Vector4;


struct Observer{
    std::vector<Vector3> observations;
    std::vector<double> pose;
    int fix_rotation;
    int fix_translation;

};

#endif /* end of include guard: COMMON_H__ */
