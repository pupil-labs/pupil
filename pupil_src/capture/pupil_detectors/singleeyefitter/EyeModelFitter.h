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
            void reset();

            // this is called with new observations from the 2D detector
            // it decides what happens ,since not all observations are added
            Detector_3D_Result update_and_detect( std::shared_ptr<Detector_2D_Result>& observation,const Detector_3D_Properties& props );


        private:

            const Vector3 mCameraCenter;
            const double mFocalLength;

            std::vector<EyeModel> mEyeModels;

            Sphere mCurrentSphere;
            Sphere mCurrentInitialSphere;

            double mPreviousPupilRadius;
            Circle mPreviousPupil;


            Contours3D unprojectObservationContours( const Contours_2D& contours) const;
            Edges3D unprojectEdges(const Edges2D& edges) const;

            void fitCircle(const Contours_2D& contours2D , const Detector_3D_Properties& props,  Detector_3D_Result& result) const;
            void filterCircle(const Edges2D& rawEdge, const Detector_3D_Properties& props,  Detector_3D_Result& result) const;

    };

}

#endif // SingleEyeFitter_h__
