
#ifndef EYEMODEL_H__
#define EYEMODEL_H__

#include "common/types.h"
#include <thread>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace singleeyefitter {

class EyeModel {

        typedef singleeyefitter::Sphere<double> Sphere;
    public:

        EyeModel(double focalLength, Vector3 cameraCenter):
            mFocalLength(std::move(focalLength)), mCameraCenter(std::move(cameraCenter)) {};

        EyeModel(const EyeModel&) = delete;
        EyeModel(EyeModel&&); // we need a explicit Move constructor because of the mutex
        ~EyeModel();

        Circle presentObservation(const ObservationPtr);

        void reset();

        // ----- Visualization --------
        std::vector<Vector3> getBinPositions(){return mBinPositions;};
        // ----- Visualization END --------

        struct CircleParams {
            double theta, psi, radius;
            CircleParams();
            CircleParams(double theta, double psi, double radius);
        };


    private:
        struct CircleDescription{
            Circle circle;
            CircleParams params;
        };

        std::vector<CircleDescription> findSphereCenter( bool use_ransac = false);
        std::vector<CircleDescription> initialiseModel( std::vector<CircleDescription>& pupils);
        void refineWithEdges( std::vector<CircleDescription>& pupils);
        void transferNewObservations();

        double getModelSupport(const Circle&  unprojectedCircle, const Circle& initialisedCircle) const;
        bool isSpatialRelevant(const Circle& circle);

        const Circle& selectUnprojectedCircle(const std::pair<const Circle, const Circle>& circles) const;
        CircleDescription getIntersectedCircle(const Circle& unprojectedCircle) const;

        Circle circleFromParams( CircleParams& params) const;
        Circle circleFromParams(const Sphere& eye,  CircleParams& params) const;



        Sphere mSphere;
        std::vector<ObservationPtr> mObservations;
        // observations are saved here and only if needed transfered to mObservation
        // since mObservations needs a mutex
        std::vector<ObservationPtr> mNewObservations;
        std::unordered_map<Vector2, bool, math::matrix_hash<Vector2>> mSpatialBins;
        std::vector<Vector3> mBinPositions; // for visualization

        std::thread mWorker;
        std::mutex mObservationMutex;
        std::mutex mSphereMutex;

        double mFocalLength;
        Vector3 mCameraCenter;

        // factor which describe how good certain properties of the model are
        double mAverageModelSupport;
        double mPerformance;
        double mMaturity;


};


} // singleeyefitter
#endif /* end of include guard: EYEMODEL_H__ */
