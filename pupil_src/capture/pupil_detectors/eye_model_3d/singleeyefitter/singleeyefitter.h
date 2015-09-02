#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry> // added by Andrew to initialize parametrized line.
#include <opencv2/core/core.hpp>
#include <singleeyefitter/cvx.h>
#include <singleeyefitter/Circle.h>
#include <singleeyefitter/Ellipse.h>
#include <singleeyefitter/Sphere.h>

namespace singleeyefitter {


    template<typename Scalar>
    inline Eigen::Matrix<Scalar, 2, 1> toEigen(const cv::Point2f& point) {
        return Eigen::Matrix<Scalar, 2, 1>(static_cast<Scalar>(point.x),
            static_cast<Scalar>(point.y));
    }
    template<typename Scalar>
    inline cv::Point2f toPoint2f(const Eigen::Matrix<Scalar, 2, 1>& point) {
        return cv::Point2f(static_cast<float>(point[0]),
            static_cast<float>(point[1]));
    }
    template<typename Scalar>
    inline cv::Point toPoint(const Eigen::Matrix<Scalar, 2, 1>& point) {
        return cv::Point(static_cast<int>(point[0]),
            static_cast<int>(point[1]));
    }
    template<typename Scalar>
    inline cv::RotatedRect toRotatedRect(const Ellipse2D<Scalar>& ellipse) {
        return cv::RotatedRect(toPoint2f(ellipse.center),
            cv::Size2f(static_cast<float>(2 * ellipse.major_radius),
            static_cast<float>(2 * ellipse.minor_radius)),
            static_cast<float>(ellipse.angle * 180 / PI));
    }
    template<typename Scalar>
    inline Ellipse2D<Scalar> toEllipse(const cv::RotatedRect& rect) {
        return Ellipse2D<Scalar>(toEigen<Scalar>(rect.center),
            static_cast<Scalar>(rect.size.width / 2),
            static_cast<Scalar>(rect.size.height / 2),
            static_cast<Scalar>(rect.angle*PI / 180));
    }

    class EyeModelFitter {
    public:
        // Typedefs
        typedef Eigen::Matrix<double, 2, 1> Vector2;
        typedef Eigen::Matrix<double, 3, 1> Vector3;
        typedef Eigen::ParametrizedLine<double, 2> Line;
        typedef Eigen::ParametrizedLine<double, 3> Line3;
        typedef singleeyefitter::Circle3D<double> Circle;
        typedef singleeyefitter::Ellipse2D<double> Ellipse;
        typedef singleeyefitter::Sphere<double> Sphere;
        typedef size_t Index;

        // Variables I use
        static const Vector3 camera_center;
        Eigen::Matrix<double, 3, 3> intrinsics;
        Sphere eye;
        Ellipse projected_eye;
        std::mutex model_mutex;
        // Model version gets incremented on initialisation/reset, so that long-running background-thread refines don't overwrite the model
        int model_version = 0;
        double scale;

        // Nonessential Variables I use
        std::vector<Line> pupil_gazelines_projection; // giving an error but will need to add in at some point
        Eigen::Matrix<double, 2,2> twoDim_A;        
        Vector2 twoDim_B;
        double count;

        // Variables I don't use, but swirski uses
        // double focal_length;
        // double region_band_width;
        // double region_step_epsilon;
        // double region_scale;

        // Constructors
        EyeModelFitter();
        EyeModelFitter(double focal_length, double x_disp, double y_disp); // used for constructing intrinsics matrix
        EyeModelFitter(double focal_length);
        // Index add_observation(Ellipse pupil);
        void add_observation(double center_x, double center_y, double major_radius, double minor_radius, double angle);
        Index add_pupil_labs_observation(Ellipse pupil);
        void reset();

        //printing & returning functions

        struct PupilParams {
            double theta, psi, radius;
            PupilParams();
            PupilParams(double theta, double psi, double radius);
        };
        struct Pupil {
            // Observation observation;
            Ellipse ellipse;
            Circle circle;
            PupilParams params;
            bool init_valid;
            std::pair<Circle, Circle> projected_circles;
            Line line;

            Pupil();
            Pupil(Ellipse ellipse, Eigen::Matrix<double,3,3> intrinsics);
        };

        std::vector<Pupil> pupils;

        // Functions used 
        void unproject_observations(double pupil_radius = 1, double eye_z = 20);
        void initialise_model();
        typedef std::function<void(const Sphere&, const std::vector<Circle>&)> CallbackFunction;
        const Circle& unproject_single_observation(Index id, double pupil_radius = 1);
        const Circle& initialise_single_observation(Index id);
        const Circle& unproject_single_observation(Pupil& pupil, double pupil_radius = 1) const;
        const Circle& initialise_single_observation(Pupil& pupil);
        Circle circleFromParams(const PupilParams& params) const;
        static Circle circleFromParams(const Sphere& eye, const PupilParams& params);

        std::vector<Vector3> intersect_contour_with_eye(std::vector<Vector2> contour);

        // functions I don't use, that require observation structure
        // const Circle& refine_single_with_contrast(Index id);
        // double single_contrast_metric(Index id) const;
        // void print_single_contrast_metric(Index id) const;
        // const Circle& refine_single_with_contrast(Pupil& pupil);
        // double single_contrast_metric(const Pupil& pupil) const;
        // void print_single_contrast_metric(const Pupil& pupil) const;
        // void refine_with_region_contrast(const CallbackFunction& callback = CallbackFunction());
        // void refine_with_inliers(const CallbackFunction& callback = CallbackFunction());

    };

}

#endif // SingleEyeFitter_h__
