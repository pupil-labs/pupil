#ifndef singleeyefitter_colors_h__
#define singleeyefitter_colors_h__

#include <opencv2/core.hpp>


namespace singleeyefitter {

    const cv::Scalar_<int> mRed_color = {0, 0, 255};
    const cv::Scalar_<int> mGreen_color = {0, 255, 0};
    const cv::Scalar_<int> mBlue_color = {255, 0, 0};
    const cv::Scalar_<int> mRoyalBlue_color = {255, 100, 100};
    const cv::Scalar_<int> mYellow_color = {255, 255, 0};
    const cv::Scalar_<int> mWhite_color = {255, 255, 255};

} // singleeyefitter namespace

#endif //singleeyefitter_types_h__
