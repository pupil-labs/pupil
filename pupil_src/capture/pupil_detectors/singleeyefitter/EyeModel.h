
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
        Circle presentObservation(const ObservationPtr);
        void reset();



    private:

        void unprojectObservations( bool use_ransac = false);
        std::vector<PupilParams> initialiseModel();
        void refineWithEdges(const std::vector<PupilParams>& pupilParams);

        double getModelSupport(const Circle&  unprojectedCircle, const Circle& initialisedCircle) const;
        bool isSpatialRelevant(const Circle& circle) const;

        const Circle& selectUnprojectedCircle(const std::pair<const Circle, const Circle>& circles) const;
        Circle getIntersectedCircle(const Circle& unprojectedCircle) const;



        Sphere mSphere;
        std::vector<ObservationPtr> mObservations;
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
