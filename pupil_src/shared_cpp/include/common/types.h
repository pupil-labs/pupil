
#ifndef singleeyefitter_types_h__
#define singleeyefitter_types_h__

#include "geometry/Ellipse.h"
#include "geometry/Circle.h"
#include "geometry/Sphere.h"
#include "projection.h"

#include <vector>
#include <memory>
#include <chrono>

#include <opencv2/core.hpp>


namespace singleeyefitter {


    //########  2D Detector ############
    typedef std::vector<std::vector<cv::Point> > Contours_2D;
    typedef std::vector<cv::Point> Contour_2D;
    typedef std::vector<cv::Point> Edges2D;
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
    typedef std::vector<Vector3> Edges3D;
    typedef std::vector<std::vector<Vector3>> Contours3D;

    struct ConfidenceValue{
        ConfidenceValue(double v,double c)
        {
            value = v;
            confidence = c;
        };
        ConfidenceValue()
        {
            value = 0;
            confidence = 0;
        };
        double value;
        double confidence;
        };

    // general time
    typedef std::chrono::steady_clock Clock;


    // every coordinates are relative to the roi
    struct Detector2DResult {
        double confidence =  0.0 ;
        Ellipse ellipse = Ellipse::Null;
        Edges2D final_edges; // edges used to fit the final ellipse in 2D
        Edges2D raw_edges;
        cv::Rect current_roi; // contains the roi for this results
        double timestamp = 0.0;
        int image_width = 0;
        int image_height = 0;

    };

    struct ModelDebugProperties{
        Sphere<double> sphere;
        Sphere<double> initialSphere;
        std::vector<Vector3> binPositions;
        double maturity;
        double solverFit;
        double confidence;
        double performance;
        double performanceGradient;
        int modelID;
        double birthTimestamp;
    };

    struct Detector3DResult {
        double confidence =  0.0 ;
        Circle circle  = Circle::Null;
        Ellipse ellipse = Ellipse::Null; // the circle projected back to 2D
        Sphere<double> sphere = Sphere<double>::Null;
        Ellipse projectedSphere = Ellipse::Null; // the sphere projected back to 2D
        double timestamp;
        int modelID = 0;
        double modelBirthTimestamp = 0.0;
        double modelConfidence = 0.0;
        //-------- For visualization ----------------
        // just valid if we want it for visualization
        Edges3D edges;
        Circle predictedCircle = Circle::Null;
        std::vector<ModelDebugProperties> models;
    };

    // use a struct for all properties and pass it to detect method every time we call it.
    // Thus we don't need to keep track if GUI is updated and cython handles conversion from Dict to struct
    struct Detector2DProperties {
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

    struct Detector3DProperties {
        float model_sensitivity;
    };

} // singleeyefitter namespace

#endif //singleeyefitter_types_h__
