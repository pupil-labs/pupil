
#ifndef singleeyefitter_types_h__
#define singleeyefitter_types_h__

#include "../Ellipse.h"
#include "../Circle.h"
#include "../Sphere.h"

#include <vector>
#include <opencv2/core/core.hpp>


namespace singleeyefitter {


    //########  2D Detector ############
    typedef std::vector<std::vector<cv::Point> > Contours_2D;
    typedef std::vector<cv::Point> Contour_2D;
    typedef std::vector<int> ContourIndices;
    typedef singleeyefitter::Ellipse2D<double> Ellipse;

    //########  3D Detector ############

    typedef Eigen::Matrix<double, 2, 1> Vector2;
    typedef Eigen::Matrix<double, 3, 1> Vector3;
    typedef Eigen::ParametrizedLine<double, 2> Line;
    typedef Eigen::ParametrizedLine<double, 3> Line3;
    typedef singleeyefitter::Circle3D<double> Circle;
    typedef size_t Index;



    // every coordinates are relative to the roi
    struct Detector_2D_Results {
        typedef singleeyefitter::Ellipse2D<double> Ellipse;
        double confidence =  0.0 ;
        Ellipse ellipse;
        Contours_2D final_contours;
        Contours_2D contours;
        std::vector<cv::Point> raw_edges;
        cv::Rect current_roi; // contains the roi for this results
        double timestamp = 0.0;
    };

    // use a struct for all properties and pass it to detect method every time we call it.
    // Thus we don't need to keep track if GUI is updated and cython handles conversion from Dict to struct
    struct Detector_2D_Properties {
        int intensity_range;
        int blur_size;
        float canny_treshold;
        float canny_ration;
        int canny_aperture;
        int pupil_size_max;
        int pupil_size_min;
        float strong_perimeter_ratio_range_min;
        float strong_perimeter_ratio_range_max;
        float strong_area_ratio_range_min;
        float strong_area_ratio_range_max;
        int contour_size_min;
        float ellipse_roundness_ratio;
        float initial_ellipse_fit_treshhold;
        float final_perimeter_ratio_range_min;
        float final_perimeter_ratio_range_max;

    };


} // singleeyefitter namespace

#endif //singleeyefitter_types_h__
