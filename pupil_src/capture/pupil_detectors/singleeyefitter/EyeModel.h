
#ifndef EYEMODEL_H__
#define EYEMODEL_H__

#include "common/types.h"
#include <thread>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <list>
#include <atomic>

namespace singleeyefitter {


    /*
        Observation class

        Hold data which is precalculated for every new observation
        Every observation is shared between different models

    */
    class Observation {
        std::shared_ptr<const Detector2DResult> mObservation2D;
        std::pair<Circle,Circle> mUnprojectedCirclePair;
        Line mProjectedCircleGaze;


    public:
        Observation(std::shared_ptr<const Detector2DResult> observation, double focalLength) :
            mObservation2D(observation)
        {
                const double circleRadius = 1.0;
                // Do a per-image unprojection of the pupil ellipse into the two fixed
                // sized circles that would project onto it. The size of the circles
                // doesn't matter here, only their center and normal does.
                mUnprojectedCirclePair = unproject(mObservation2D->ellipse, circleRadius , focalLength);
                 // Get projected circles and gaze vectors
                //
                // Project the circle centers and gaze vectors down back onto the image
                // plane. We're only using them as line parametrisations, so it doesn't
                // matter which of the two centers/gaze vectors we use, as the
                // two gazes are parallel and the centers are co-linear.
                const auto& c = mUnprojectedCirclePair.first.center;
                const auto& v = mUnprojectedCirclePair.first.normal;
                Vector2 cProj = project(c, focalLength);
                Vector2 vProj = project(v + c, focalLength) - cProj;
                vProj.normalize();
                mProjectedCircleGaze = Line(cProj, vProj);

        }
        Observation( const Observation& that ) = delete; // forbid copying
        Observation( Observation&& that ) = delete; // forbid moving
        Observation() = delete; // forbid default construction
        const std::shared_ptr<const Detector2DResult> getObservation2D() const { return mObservation2D;};
        const std::pair<Circle,Circle>& getUnprojectedCirclePair() const { return mUnprojectedCirclePair; };
        const Line& getProjectedCircleGaze() const { return mProjectedCircleGaze; };

    };

    typedef std::shared_ptr<const Observation> ObservationPtr;


class EyeModel {

        typedef singleeyefitter::Sphere<double> Sphere;
    public:

        EyeModel( int modelId, Clock::time_point timestamp,  double focalLength, Vector3 cameraCenter, int initialUncheckedPupils = 3, double binResolution = 0.05 , int filterwindowSize = 200): //TODO should the filter size depend on the framerate ?
           mModelID(modelId), mTimestamp(timestamp), mTotalBins(std::pow(std::floor(1.0/binResolution), 2 ) * 4 ),  mFilterWindowSize(filterwindowSize), mBinResolution(binResolution), mInitialUncheckedPupils(initialUncheckedPupils), mFocalLength(std::move(focalLength)), mCameraCenter(std::move(cameraCenter)) { };

        EyeModel(const EyeModel&) = delete;
        //EyeModel(EyeModel&&); // we need a explicit 1/Move constructor because of the mutex
        ~EyeModel();



        Circle presentObservation(const ObservationPtr);
        Sphere getSphere() const;
        Sphere getInitialSphere() const;

        // Describing how good different properties of the Eye are
        double getMaturity() const ; // How much spatial variance there is
        double getPerformance() const; // The average of the model support
        double getFit() const ; // The residual of the sphere calculation

        int getModelID() const { return mModelID; };
        Clock::time_point getTimestamp() const { return mTimestamp; };

        // ----- Visualization --------
        std::vector<Vector3> getBinPositions() const {return mBinPositions;};
        // ----- Visualization END --------


    private:


        struct PupilParams {
            double theta, psi, radius;
            PupilParams() : theta(0), psi(0), radius(0) {};
            PupilParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius){};
        };

        struct Pupil{
            Circle mCircle;
            PupilParams mParams;
            const ObservationPtr mObservationPtr;
            Pupil( const ObservationPtr observationPtr ) : mObservationPtr( observationPtr ){};
        };


        Sphere findSphereCenter( bool use_ransac = true);
        Sphere initialiseModel();
        double refineWithEdges( Sphere& sphere  );
        bool tryTransferNewObservations();

        double getModelSupport(const Circle&  unprojectedCircle, const Circle& initialisedCircle) const;
        bool isSpatialRelevant(const Circle& circle);

        const Circle& selectUnprojectedCircle(const Sphere& sphere, const std::pair<const Circle, const Circle>& circles) const;
        void initialiseSingleObservation( const Sphere& sphere, Pupil& pupil) const;
        Circle getIntersectedCircle( const Sphere& sphere, const Circle& circle) const;

        //Circle circleFromParams( CircleParams& params) const;
        Circle circleFromParams(const Sphere& eye,const  PupilParams& params) const;

        void calculatePerformance( const Circle& unprojectedCircle , const Circle& intersectedCircle);


        std::unordered_map<Vector2, bool, math::matrix_hash<Vector2>> mSpatialBins;
        std::vector<Vector3> mBinPositions; // for visualization

        mutable std::mutex mModelMutex;
        std::mutex mPupilMutex;
        std::thread mWorker;


        // Factors which describe how good certain properties of the model are
        double mFit; // Residual of Ceres sovler
        double mPerformance; // Average model support
        double mMaturity; // bin amounts

        const double mFocalLength;
        const Vector3 mCameraCenter;
        const int mInitialUncheckedPupils;
        const double mBinResolution;
        const int mTotalBins;
        const int mFilterWindowSize; // Window size of the average moving filter for mPerformance
        const int mModelID;
        const Clock::time_point mTimestamp;

        std::list<double> mModelSupports; // values to calculate the average

        // Thread sensitive variables
        Sphere mSphere;
        Sphere mInitialSphere;
        std::vector<Pupil> mSupportingPupils;

        std::atomic<int> mSupportingPupilSize; // use this to get the SupportedPupil size

        // observations are saved here and only if needed transfered to mObservation
        // since mObservations needs a mutex
        std::vector<Pupil> mSupportingPupilsToAdd;
};


} // singleeyefitter
#endif /* end of include guard: EYEMODEL_H__ */
