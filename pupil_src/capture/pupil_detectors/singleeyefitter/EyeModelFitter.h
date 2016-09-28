#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <opencv2/video/tracking.hpp> // Kalaman Filter

#include <vector>
#include <memory>
#include <Eigen/Core>

#include "common/types.h"
#include "ImageProcessing/cvx.h"
#include "geometry/Circle.h"
#include "geometry/Ellipse.h"
#include "geometry/Sphere.h"
#include "EyeModel.h"

#include "logger/pycpplogger.h"


namespace singleeyefitter {


    class EyeModelFitter {
        public:

            typedef singleeyefitter::Sphere<double> Sphere;
            typedef std::unique_ptr<EyeModel> EyeModelPtr;

            // Constructors
            EyeModelFitter(double focalLength, Vector3 cameraCenter = Vector3::Zero() );
            EyeModelFitter() = delete;


            double getFocalLength(){ return mFocalLength; };
            void reset();

            // this is called with new observations from the 2D detector
            // it decides what happens ,since not all observations are added
            Detector3DResult updateAndDetect( std::shared_ptr<Detector2DResult>& observation,const Detector3DProperties& props, bool debug = false );


        private:

            const Vector3 mCameraCenter;
            const double mFocalLength;

            bool mDebug;

            Clock::time_point mLastTimeModelAdded, mLastTimePerformancePenalty;

            int mNextModelID;
            std::unique_ptr<EyeModel> mActiveModelPtr;
            std::list<EyeModelPtr> mAlternativeModelsPtrs;

            Sphere mCurrentSphere;
            Sphere mCurrentInitialSphere;

            cv::KalmanFilter mPupilState;

            double mLastFrameTimestamp; //needed to calculate framerate
            int mApproximatedFramerate;
            math::SMA<double> mAverageFramerate;

            pupillabs::PyCppLogger mLogger;

            void checkModels( float sensitivity,double frame_timestamp);

            //Contours3D unprojectContours( const Contours_2D& contours) const;
            Edges3D unprojectEdges(const Edges2D& edges) const;

            // whenever the 2D fit is bad we wanna call this and predict an new circle to use for findCircle
            Circle predictPupilState( double deltaTime );
            Circle correctPupilState( const Circle& circle );
            double getPupilPositionErrorVar () const;
            double getPupilSizeErrorVar() const;


            //void fitCircle(const Contours_2D& contours2D , const Detector3DProperties& props,  Detector3DResult& result) const;
            void filterCircle(const Edges2D& rawEdge, const Detector3DProperties& props,  Detector3DResult& result) const;
            void filterCircle2( const Circle& predictedCircle, const Edges2D& rawEdge, const Detector3DProperties& props,  Detector3DResult& result) const;
            void filterCircle3( const Circle& predictedCircle, const Edges2D& rawEdge, const Detector3DProperties& props,  Detector3DResult& result) const;

    };

}

#endif // SingleEyeFitter_h__
