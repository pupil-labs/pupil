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

            struct PupilParams {
                double theta, psi, radius;
                PupilParams();
                PupilParams(double theta, double psi, double radius);
            };
            struct Pupil {
                //Observation observation;
                std::shared_ptr<Detector_2D_Results> observation;
                Circle circle;
                Circle unprojected_circle; // we gonna keep track of the unprojected circle, because we need these parameters later
                double fit_goodness;
                PupilParams params;
                bool init_valid;
                bool processed; // indicate if this pupil is already processed
                Pupil();
                Pupil(std::shared_ptr<Detector_2D_Results> observation);
                //Pupil(Ellipse ellipse);
            };

            // Constructors
            EyeModelFitter();
            EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon);

            // Index add_observation(cv::Mat image, Ellipse pupil, int n_pseudo_inliers = 0);
            // Index add_observation(cv::Mat image, Ellipse pupil, std::vector<cv::Point2f> pupil_inliers);
            //Index add_observation( Ellipse pupil );
            Index add_observation( const Pupil& pupil );
            /*
                contours contains pointers to memory location of each contour
                contour_sizes contains the size of the corresponded contour, so we know how much memory we can claim on the c++ side
            */
            //Index add_observation(Ellipse pupil, std::vector<int32_t*> contour_ptrs ,  std::vector<size_t> contour_sizes);
            void reset();

            //
            // Global (eye+pupils) calculations
            //

            void unproject_observations(double pupil_radius = 5, double eye_z = 57, bool use_ransac = true);

            void initialise_model();


            typedef std::function<void(const Sphere&, const std::vector<Circle>&)> CallbackFunction;

            void refine_with_region_contrast(const CallbackFunction& callback = CallbackFunction());

            void refine_with_inliers(const CallbackFunction& callback = CallbackFunction());

            //
            // Pubil-Laps addons
            //

            // this is called with new observations from the 2D detector
            // it decides what happens ,since not all observations are added
            void update( std::shared_ptr<Detector_2D_Results>& observation, Detector_3D_Properties& props );

            //void unproject_last_raw_edges();

           // void fit_circle_for_last_contour();

        //private:

            bool spatial_variance_check( const Circle&  circle );
            bool model_support_check( const Circle&  unprojected_circle, const Circle& initialised_circle );


            //
            // Local (single pupil) calculations
            //
            const Circle& unproject_single_observation(Index id, double pupil_radius = 5);
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

            // data we get each frame
            Contours3D eye_contours;
            Circle latest_pupil_circle;
            std::vector<Vector3> edges; // just for visualization
            Contours3D final_circle_contours; // just for visualiziation, contains all points which fit best the circle
            std::vector<Contours3D> final_candidate_contours; // just for visualiziation, contains all contours which are a candidate for the fit
            Vector3 gaze_vector;

            void unproject_observation_contours( const Contours_2D& contours);
            //void unproject_last_raw_edges();
            void fit_circle_for_eye_contours( float max_residual = 20, float max_variance = 0.7, float min_radius = 2, float max_radius = 4 );


            std::unordered_map<Vector2, bool, math::matrix_hash<Vector2>> pupil_position_bins;
            std::vector<Vector3> bin_positions; // for visualization

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
