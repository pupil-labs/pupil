
#ifndef singleeyefitter_types_h__
#define singleeyefitter_types_h__

#include "../Geometry/Ellipse.h"
#include "../Geometry/Circle.h"
#include "../Geometry/Sphere.h"
#include "../projection.h"

#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>


namespace singleeyefitter {


    //########  2D Detector ############
    typedef std::vector<std::vector<cv::Point> > Contours_2D;
    typedef std::vector<cv::Point> Contour_2D;
    typedef std::vector<int> ContourIndices;
    typedef Ellipse2D<double> Ellipse;

    //########  3D Detector ############

    typedef Eigen::Matrix<double, 2, 1> Vector2;
    typedef Eigen::Matrix<double, 3, 1> Vector3;
    typedef Eigen::ParametrizedLine<double, 2> Line;
    typedef Eigen::ParametrizedLine<double, 3> Line3;
    typedef Circle3D<double> Circle;
    typedef size_t Index;

    typedef std::vector<Vector3> Contour3D;
    typedef std::vector<std::vector<Vector3>> Contours3D;


    // every coordinates are relative to the roi
    struct Detector_2D_Result {
        double confidence =  0.0 ;
        Ellipse ellipse = Ellipse::Null;
        //Contours_2D final_contours;
        Contours_2D contours;
        std::vector<cv::Point> final_edges; // edges used to fit the final ellipse in 2D
        //std::vector<cv::Point> raw_edges;
        cv::Rect current_roi; // contains the roi for this results
        double timestamp = 0.0;
        int image_width = 0;
        int image_height = 0;

    };

    struct Detector_3D_Result {
        double confidence =  0.0 ;
        Circle circle  = Circle::Null;
        Ellipse ellipse = Ellipse::Null; // the circle projected back to 2D
        double fitGoodness =  -1.0;
        double timestamp = 0.0;
        //-------- For visualization ----------------
        // just valid if we want it for visualization
        Contours3D contours;
        Contours3D fittedCircleContours;
        Sphere<double> sphere;
        std::vector<Vector3> binPositions;
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
        float ellipse_true_support_min_dist;

    };
    struct Detector_3D_Properties {
        float max_fit_residual;
        float max_circle_variance;
        float pupil_radius_min;
        float pupil_radius_max;
        int   combine_evaluation_max;
        int   combine_depth_max;

    };

} // singleeyefitter namespace

#endif //singleeyefitter_types_h__
