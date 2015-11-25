#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

//#include <opencv2/core/core.hpp>
#include "common/types.h"
#include "ImageProcessing/cvx.h"
#include "Geometry/Circle.h"
#include "Geometry/Ellipse.h"
#include "Geometry/Sphere.h"
#include "EyeModel.h"

#include <vector>
#include <Eigen/Core>

namespace singleeyefitter {


    class EyeModelFitter {
        public:

            typedef singleeyefitter::Sphere<double> Sphere;

            // Constructors
            EyeModelFitter(double focalLength, Vector3 cameraCenter = Vector3::Zero() );
            EyeModelFitter() = delete;


            double getFocalLength(){ return mFocalLength; };
            Sphere getSphere(){ return mCurrentSphere; };

            void reset();

            // this is called with new observations from the 2D detector
            // it decides what happens ,since not all observations are added
            Detector_3D_Result update_and_detect( std::shared_ptr<Detector_2D_Result>& observation,const Detector_3D_Properties& props );


        private:
            //const Circle& selectUnprojectedCircle( sphere, focal_length, std::pair<Circle,Circle> );
            //const Circle& getIntersectedCircle(sphere, camera, unprojectedCircle  );


            const Vector3 mCameraCenter;
            const double mFocalLength;

            std::vector<EyeModel> mEyeModels;

            Sphere mCurrentSphere;
            Circle mLatestPupil;
            double mPreviousPupilRadius;

            // data we get each frame
            Contours3D mContoursOnSphere;
            std::vector<Vector3> edges; // just for visualization
            Contours3D final_circle_contours; // just for visualiziation, contains all points which fit best the circle
            std::vector<Contours3D> final_candidate_contours; // just for visualiziation, contains all contours which are a candidate for the fit
            Vector3 gaze_vector;

            void unproject_observation_contours( const Contours_2D& contours);
            //void unproject_last_raw_edges();
            double fit_circle_for_eye_contours( const Detector_3D_Properties& props);


            Circle circleFromParams(const PupilParams& params) const;
            static Circle circleFromParams(const Sphere& eye, const PupilParams& params);
    };

}

#endif // SingleEyeFitter_h__
