#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <mutex>
#include <deque>
#include <unordered_map>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "common/types.h"
#include "mathHelper.h"
#include "ImageProcessing/cvx.h"
#include "Geometry/Circle.h"
#include "Geometry/Ellipse.h"
#include "Geometry/Sphere.h"

namespace singleeyefitter {


    class EyeModelFitter {
        public:

            typedef singleeyefitter::Sphere<double> Sphere;

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
            //Index add_observation( Ellipse pupil );
            Index add_observation( std::shared_ptr<Detector_2D_Results>& observation , int image_width = 640, int image_height = 480, bool convert_to_eyefitter_space = true );
            /*
                contours contains pointers to memory location of each contour
                contour_sizes contains the size of the corresponded contour, so we know how much memory we can claim on the c++ side
            */
            //Index add_observation(Ellipse pupil, std::vector<int32_t*> contour_ptrs ,  std::vector<size_t> contour_sizes);
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

            void unproject_last_contour();
            void unproject_last_raw_edges();

           // void fit_circle_for_last_contour();
            void fit_circle_for_last_contour( float max_residual = 20, float max_variance = 0.7, float min_radius = 2, float max_radius = 4 );

            // struct Observation {
            //     //cv::Mat image;
            //     Ellipse ellipse;
            //     //std::vector<cv::Point2f> inliers;
            //     std::vector<std::vector<cv::Point2i>> contours;
            //     //std::vector<std::vector<int32_t>> contours;
            //     Observation();
            //     Observation(/*cv::Mat image,*/ Ellipse ellipse/*, std::vector<cv::Point2f> inliers*/,  std::vector<std::vector<cv::Point2i>> contours);
            //     Observation( std::shared_ptr<Detector_2D_Results> observation, );
            // };

            struct PupilParams {
                double theta, psi, radius;
                PupilParams();
                PupilParams(double theta, double psi, double radius);
            };
            struct Pupil {
                //Observation observation;
                std::shared_ptr<Detector_2D_Results> observation;
                Contours3D contours;
                std::vector<Vector3> edges;
                Contours3D final_circle_contours; // just for visualiziation, contains all points which fit best the circle
                Circle circle; // this one is the unprojected circle
                Circle circle_fitted;  // this is the circle fitted form the unprojectd contours
                double fit_goodness;
                PupilParams params;
                bool init_valid;
                bool processed; // indicate if this pupil is already processed
                Pupil();
                Pupil(std::shared_ptr<Detector_2D_Results> observation);
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
            std::deque<Pupil> pupils;
            size_t max_pupils;   // this are the max pupils we wanna consider for calculations
            std::mutex model_mutex;
            // Model version gets incremented on initialisation/reset, so that long-running background-thread refines don't overwrite the model
            int model_version = 0;

            // Mean of all pupil observations, used to calculate variance of a new observation
            double psi_mean = 0.0;
            double theta_mean = 0.0;


            // in order to check if new observations are unique  (not in the same area as previous one )
            // the position on the speher (only x,y coords) are binned  (spatial binning) an inserted into the right bin
            // !! this is not a correct method to check if they are uniformly distibuted, because if x,y are uniformly distibuted
            // it doesn't mean points on the spehre are uniformly distibuted
            // to uniformly distibute points on a sphere you have to check if the area is equal on the sphere of two bins
            // and we just look on have the sphere, it's like projecting a checkboard grid on the sphere
            std::unordered_map<Vector2, bool, math::matrix_hash<Vector2>> pupil_position_bins;
            std::vector<Vector3> bin_positions; // for debuging

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
