#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <mutex>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "cvx.h"
#include "Circle.h"
#include "Ellipse.h"
#include "Sphere.h"

namespace singleeyefitter {


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

            static const Vector3 camera_center;

            // Public fields
            double focal_length;
            double region_band_width;
            double region_step_epsilon;
            double region_scale;

            // Constructors
            EyeModelFitter();
            EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon);

            // Index add_observation(cv::Mat image, Ellipse pupil, int n_pseudo_inliers = 0);
            // Index add_observation(cv::Mat image, Ellipse pupil, std::vector<cv::Point2f> pupil_inliers);
            Index add_observation( Ellipse pupil );

            /*
                contours contains pointers to memory location of each contour
                contour_sizes contains the size of the corresponded contour, so we know how much memory we can claim on the c++ side
            */
            Index add_observation(Ellipse pupil, std::vector<int32_t*> contour_ptrs ,  std::vector<size_t> contour_sizes);
            void reset();

            //
            // Global (eye+pupils) calculations
            //

            void unproject_observations(double pupil_radius = 1, double eye_z = 20, bool use_ransac = true);

            void initialise_model();


            typedef std::function<void(const Sphere&, const std::vector<Circle>&)> CallbackFunction;

            void refine_with_region_contrast(const CallbackFunction& callback = CallbackFunction());

            void refine_with_inliers(const CallbackFunction& callback = CallbackFunction());

            //
            // Pubil-Laps addons
            //

            void unproject_contours();
            void unwrap_contours();

            struct Observation {
                //cv::Mat image;
                Ellipse ellipse;
                //std::vector<cv::Point2f> inliers;
                std::vector<std::vector<cv::Point2i>> contours;
                //std::vector<std::vector<int32_t>> contours;
                Observation();
                Observation(/*cv::Mat image,*/ Ellipse ellipse/*, std::vector<cv::Point2f> inliers*/,  std::vector<std::vector<cv::Point2i>> contours);
            };

            struct PupilParams {
                double theta, psi, radius;
                PupilParams();
                PupilParams(double theta, double psi, double radius);
            };
            struct Pupil {
                Observation observation;
                std::vector<std::vector<Vector3>> unprojected_contours;  // whre to put this ? observations ? keep projected contours ?
                std::vector<std::vector<Vector2>> unwrapped_contours;
                Circle circle;
                PupilParams params;
                bool init_valid;
                bool processed; // indicate if this pupil is already processed
                Pupil();
                Pupil(Observation observation);
                //Pupil(Ellipse ellipse);
            };

            //
            // Local (single pupil) calculations
            //
            const Circle& unproject_single_observation(Index id, double pupil_radius = 1);
            const Circle& initialise_single_observation(Index id);
            //const Circle& refine_single_with_contrast(Index id);
            //double single_contrast_metric(Index id) const;
            //void print_single_contrast_metric(Index id) const;

            Sphere eye;
            std::vector<Pupil> pupils;
            std::mutex model_mutex;
            // Model version gets incremented on initialisation/reset, so that long-running background-thread refines don't overwrite the model
            int model_version = 0;

            const Circle& unproject_single_observation(Pupil& pupil, double pupil_radius = 1) const;
            const Circle& initialise_single_observation(Pupil& pupil);
            // const Circle& refine_single_with_contrast(Pupil& pupil);
            // double single_contrast_metric(const Pupil& pupil) const;
            // void print_single_contrast_metric(const Pupil& pupil) const;

            Circle circleFromParams(const PupilParams& params) const;
            static Circle circleFromParams(const Sphere& eye, const PupilParams& params);
    };

}

#endif // SingleEyeFitter_h__
