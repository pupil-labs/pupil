#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include "common/types.h"
#include "ImageProcessing/cvx.h"
#include "Geometry/Circle.h"
#include "Geometry/Ellipse.h"
#include "Geometry/Sphere.h"
#include "EyeModel.h"

#include <vector>
#include <memory>
#include <Eigen/Core>


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


            double mLastFrameTimestamp; //needed to calculate framerate
            int mApproximatedFramerate;
            math::SMA<double> mAverageFramerate;

            void checkModels();

            //Contours3D unprojectContours( const Contours_2D& contours) const;
            Edges3D unprojectEdges(const Edges2D& edges) const;

            //void fitCircle(const Contours_2D& contours2D , const Detector3DProperties& props,  Detector3DResult& result) const;
            void filterCircle(const Edges2D& rawEdge, const Detector3DProperties& props,  Detector3DResult& result) const;
            void filterCircle2( const Circle& predictedCircle, const Edges2D& rawEdge, const Detector3DProperties& props,  Detector3DResult& result) const;

    };

}

#endif // SingleEyeFitter_h__
