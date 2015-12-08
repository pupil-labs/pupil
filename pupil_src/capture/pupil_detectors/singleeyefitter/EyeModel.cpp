

#include "EyeModel.h"

#include <algorithm>
#include <future>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>

#include "EllipseDistanceApproxCalculator.h"
#include "EllipseGoodnessFunction.h"
#include "EllipseDistanceResidualFunction.h"
#include "EllipsePointDistanceFunction.h"

#include "CircleDeviationVariance3D.h"
#include "CircleEvaluation3D.h"
#include "CircleGoodness3D.h"

#include "utils.h"
#include "intersect.h"
#include "projection.h"
#include "fun.h"

#include "mathHelper.h"
#include "distance.h"




namespace singleeyefitter {



// EyeModel::EyeModel(EyeModel&& that) :
//     mInitialUncheckedPupils(that.mInitialUncheckedPupils), mFocalLength(that.mFocalLength), mCameraCenter(that.mCameraCenter),
//     mTotalBins(that.mTotalBins), mBinResolution(that.mBinResolution), mFilterWindowSize(that.mFilterWindowSize), mTimestamp(that.mTimestamp ),
//     mModelID(that.mModelID)
// {
//     std::lock_guard<std::mutex> lock(that.mModelMutex);
//     mSupportingPupils = std::move(that.mSupportingPupils);
//     mSupportingPupilsToAdd = std::move(that.mSupportingPupilsToAdd);
//     mSphere = std::move(that.mSphere);
//     mInitialSphere = std::move(that.mInitialSphere);
//     mSpatialBins = std::move(that.mSpatialBins);
//     mBinPositions = std::move(that.mBinPositions);
//     mFit = std::move(that.mFit);
//     mPerformance = std::move(that.mPerformance);
//     mMaturity = std::move(that.mMaturity);
//     mModelSupports = std::move(that.mModelSupports);
//     mSupportingPupilSize = mSupportingPupils.size();
//     std::cout << "MOVE EYE MODEL" << std::endl;
// }

// EyeModel& EyeModel::operator=(EyeModel&& that){

//     std::lock_guard<std::mutex> lock(that.mModelMutex);
//     mSupportingPupils = std::move(that.mSupportingPupils);
//     mSupportingPupilsToAdd = std::move(that.mSupportingPupilsToAdd);
//     mSphere = std::move(that.mSphere);
//     mInitialSphere = std::move(that.mInitialSphere);
//     mSpatialBins = std::move(that.mSpatialBins);
//     mBinPositions = std::move(that.mBinPositions);
//     mFit = std::move(that.mFit);
//     mPerformance = std::move(that.mPerformance);
//     mMaturity = std::move(that.mMaturity);
//     mModelSupports = std::move(that.mModelSupports);
//     mSupportingPupilSize = mSupportingPupils.size();
//     std::cout << "MOVE ASSGINE EYE MODEL" << std::endl;
//     return *this;
// }

EyeModel::~EyeModel(){

    //wait for thread to finish before we dealloc
    if( mWorker.joinable() )
        mWorker.join();
}


Circle EyeModel::presentObservation(const ObservationPtr newObservationPtr)
{


    Circle circle;
    bool shouldAddObservation = false;

    //Check for properties if it's a candidate we can use
    if (mSphere != Sphere::Null && (mSupportingPupilSize + mSupportingPupilsToAdd.size()) >= mInitialUncheckedPupils ) {

        // select the right circle depending on the current model
        const Circle& unprojectedCircle = selectUnprojectedCircle(mSphere, newObservationPtr->getUnprojectedCirclePair() );

        if (isSpatialRelevant(unprojectedCircle)) {
            shouldAddObservation = true;
        } else {
            //std::cout << " spatial check failed"  << std::endl;
        }

        // initialised circle. circle parameters addapted to our current eye model
        circle = getIntersectedCircle(mSphere, unprojectedCircle);
        calculatePerformance( unprojectedCircle, circle );

        if (circle == Circle::Null)
            circle = unprojectedCircle; // at least return the unprojected circle


    } else { // no valid sphere yet
        std::cout << "add without check" << std::endl;
        shouldAddObservation = true;
    }

    if (shouldAddObservation) {
        //std::cout << "add" << std::endl;
        //if the observation passed all tests we can add it
        mSupportingPupilsToAdd.emplace_back( newObservationPtr );

    }

   if( mSupportingPupilsToAdd.size() > 3  ){

            if(tryTransferNewObservations() ) {

                auto work = [&](){
                    std::lock_guard<std::mutex> lockPupil(mPupilMutex);
                    auto sphere  = initialiseModel();
                    auto sphere2 = sphere;
                    double fit = refineWithEdges(sphere);
                    {
                        std::lock_guard<std::mutex> lockModel(mModelMutex);
                        mInitialSphere = sphere2;
                        mSphere = sphere;
                        mFit = fit;
                    }
                 };
                // needed in order to assign a new thread
                if( mWorker.joinable() )
                    mWorker.join(); // we should never wait here because tryTransferNewObservations is false if the work isn't finished

                mWorker = std::thread(work);
                //work();
            }
     }

    return circle;
}

EyeModel::Sphere EyeModel::findSphereCenter( bool use_ransac /*= true*/)
{
    using math::sq;

    Sphere sphere;

    if (mSupportingPupils.size() < 2) {
        return Sphere::Null;
    }
    const double eyeZ = 57; // could be any value

    // should we save them some where else ?
    std::vector<const Line> pupilGazelinesProjected;
    for (const auto& pupil : mSupportingPupils) {
        pupilGazelinesProjected.push_back( pupil.mObservationPtr->getProjectedCircleGaze() );
    }

    // Get eyeball center
    //
    // Find a least-squares 'intersection' (point nearest to all lines) of
    // the projected 2D gaze vectors. Then, unproject that circle onto a
    // point a fixed distance away.
    //
    // For robustness, use RANSAC to eliminate stray gaze lines
    //
    // (This has to be done here because it's used by the pupil circle
    // disambiguation)
    Vector2 eyeCenterProjected;
    bool validEye;

    if ( use_ransac ) {
        auto indices = fun::range_<std::vector<size_t>>(pupilGazelinesProjected.size());
        const int n = 2;
        double w = 0.3;
        double p = 0.9999;
        int k = ceil(log(1 - p) / log(1 - pow(w, n)));
        double epsilon = 10;
        // auto huber_error = [&](const Vector2 & point, const Line & line) {
        //     double dist = euclidean_distance(point, line);

        //     if (sq(dist) < sq(epsilon))
        //         return sq(dist) / 2;
        //     else
        //         return epsilon * (abs(dist) - epsilon / 2);
        // };
        auto error = [&](const Vector2 & point, const Line & line) {
            double dist = euclidean_distance(point, line);

            if (sq(dist) < sq(epsilon))
                return sq(dist);
            else
                return sq(epsilon);
        };
        auto bestInlierIndices = decltype(indices)();
        Vector2 bestEyeCenterProjected;// = nearest_intersect(pupilGazelinesProjected);
        double bestLineDistanceError = std::numeric_limits<double>::infinity();// = fun::sum(LAMBDA(const Line& line)(error(bestEyeCenterProjected,line)), pupilGazelinesProjected);

        for (int i = 0; i < k; ++i) {
            auto indexSample = singleeyefitter::randomSubset(indices, n);
            auto sample = fun::map([&](size_t i) { return pupilGazelinesProjected[i]; }, indexSample);
            auto sampleCenterProjected = nearest_intersect(sample);
            auto indexInliers = fun::filter(
            [&](size_t i) { return euclidean_distance(sampleCenterProjected, pupilGazelinesProjected[i]) < epsilon; },
            indices);
            auto inliers = fun::map([&](size_t i) { return pupilGazelinesProjected[i]; }, indexInliers);

            if (inliers.size() <= w * pupilGazelinesProjected.size()) {
                continue;
            }

            auto inlierCenterProj = nearest_intersect(inliers);
            double lineDistanceError = fun::sum(
            [&](size_t i) { return error(inlierCenterProj, pupilGazelinesProjected[i]); },
            indices);

            if (lineDistanceError < bestLineDistanceError) {
                bestEyeCenterProjected = inlierCenterProj;
                bestLineDistanceError = lineDistanceError;
                bestInlierIndices = std::move(indexInliers);
            }
        }

        // std::cout << "Inliers: " << bestInlierIndices.size()
        //     << " (" << (100.0*bestInlierIndices.size() / pupilGazelinesProjected.size()) << "%)"
        //     << " = " << bestLineDistanceError
        //     << std::endl;

        if (bestInlierIndices.size() > 0) {
            eyeCenterProjected = bestEyeCenterProjected;
            validEye = true;

        } else {
            validEye = false;
        }

    } else {

        eyeCenterProjected = nearest_intersect(pupilGazelinesProjected);
        validEye = true;
    }

    if (validEye) {
        sphere.center << eyeCenterProjected* eyeZ / mFocalLength,
                   eyeZ;
        sphere.radius = 1;

        // Disambiguate pupil circles using projected eyeball center
        //
        // Assume that the gaze vector points away from the eye center, and
        // so projected gaze points away from projected eye center. Pick the
        // solution which satisfies this assumption

        for (size_t i = 0; i < mSupportingPupils.size(); ++i) {
            const auto& pupilPair = mSupportingPupils[i].mObservationPtr->getUnprojectedCirclePair();
            const auto& line = mSupportingPupils[i].mObservationPtr->getProjectedCircleGaze();
            const auto& originProjected = line.origin();
            const auto& directionProjected = line.direction();

            // Check if directionProjected going away from est eye center. If it is, then
            // the first circle was correct. Otherwise, take the second one.
            // The two normals will point in opposite directions, so only need
            // to check one.
            if ((originProjected - eyeCenterProjected).dot(directionProjected) >= 0) {
                mSupportingPupils[i].mCircle =  pupilPair.first;

            } else {
                mSupportingPupils[i].mCircle = pupilPair.second;
            }

            // calculate the center variance of the projected gaze vectors to the current eye center
          //  center_distance_variance += euclidean_distance_squared( eye.center, Line3(pupils[i].circle.center, pupils[i].circle.normal ) );

        }
        //center_distance_variance /= pupils.size();
        //std::cout << "center distance variance " << center_distance_variance << std::endl;

    } else {
        // No inliers, so no eye
        sphere = Sphere::Null;
    }

    return sphere;

}

EyeModel::Sphere EyeModel::initialiseModel(){


    Sphere sphere = findSphereCenter();

    if (sphere == Sphere::Null) {
        return sphere;
    }
    std::cout << "init model" << std::endl;

    // Find pupil positions on eyeball to get radius
    //
    // For each image, calculate the 'most likely' position of the pupil
    // circle given the eyeball sphere estimate and gaze vector. Re-estimate
    // the gaze vector to be consistent with this position.
    // First estimate of pupil center, used only to get an estimate of eye radius
    double eyeRadiusAcc = 0;
    int eyeRadiusCount = 0;

    for (const auto& pupil : mSupportingPupils) {

        // Intersect the gaze from the eye center with the pupil circle
        // center projection line (with perfect estimates of gaze, eye
        // center and pupil circle center, these should intersect,
        // otherwise find the nearest point to both lines)
        Vector3 pupilCenter = nearest_intersect(Line3(sphere.center, pupil.mCircle.normal),
                               Line3(mCameraCenter, pupil.mCircle.center.normalized()));
        auto distance = (pupilCenter - sphere.center).norm();
        eyeRadiusAcc += distance;
        ++eyeRadiusCount;
    }

    // Set the eye radius as the mean distance from pupil centers to eye center
    sphere.radius = eyeRadiusAcc / eyeRadiusCount;

    // Second estimate of pupil radius, used to get position of pupil on eye

    //TODO do we really need this if we don't do refinement ?????
    for ( auto& pupil : mSupportingPupils) {
        initialiseSingleObservation(sphere, pupil);
    }

    // Scale eye to anthropomorphic average radius of 12mm
    auto scale = 12.0 / sphere.radius;
    sphere.radius = 12.0;
    sphere.center *= scale;
    double center_distance_variance = 0;
    for ( auto& pupil : mSupportingPupils) {
        pupil.mParams.radius *= scale;
        pupil.mCircle = circleFromParams(sphere, pupil.mParams);
         // calculate the center variance of the projected gaze vectors to the current eye center
        //center_distance_variance += euclidean_distance_squared( sphere.center, Line3(pupil.circle.center, pupil.unprojected_circle.normal ) );
    }

    //center_distance_variance /= eyeRadiusCount;
    //std::cout << "center distance variance " << center_distance_variance << std::endl;
    return sphere;

}

double EyeModel::refineWithEdges(Sphere& sphere )
{
    int current_model_version;
    Eigen::Matrix<double, Eigen::Dynamic, 1> x;
    x = Eigen::Matrix<double, Eigen::Dynamic, 1>(3 + 3 * mSupportingPupils.size());
    x.segment<3>(0) = sphere.center;
    for (int i = 0; i < mSupportingPupils.size(); ++i) {
        const PupilParams& pupilParams = mSupportingPupils[i].mParams;
        x.segment<3>(3 + 3 * i)[0] = pupilParams.theta;
        x.segment<3>(3 + 3 * i)[1] = pupilParams.psi;
        x.segment<3>(3 + 3 * i)[2] = pupilParams.radius;
    }


    ceres::Problem problem;
    {
        for (int i = 0; i < mSupportingPupils.size(); ++i) {
           /* const cv::Mat& eye_image = pupils[i].observation.image;*/
            const auto& pupilInliers = mSupportingPupils[i].mObservationPtr->getObservation2D()->final_edges;

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<EllipseDistanceResidualFunction<double>, ceres::DYNAMIC, 3, 3>(
                new EllipseDistanceResidualFunction<double>(/*eye_image,*/ pupilInliers, sphere.radius, mFocalLength),
                pupilInliers.size()
                ),
                NULL, &x[0], &x[3 + 3 * i]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 400;
    options.function_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = false;
    options.update_state_every_iteration = false;
    // if (callback) {
    //     struct CallCallbackWrapper : public ceres::IterationCallback
    //     {
    //         double eye_radius;
    //         const CallbackFunction& callback;
    //         const Eigen::Matrix<double, Eigen::Dynamic, 1>& x;

    //         CallCallbackWrapper(const EyeModelFitter& fitter, const CallbackFunction& callback, const Eigen::Matrix<double, Eigen::Dynamic, 1>& x)
    //             : eye_radius(fitter.eye.radius), callback(callback), x(x) {}

    //         virtual ceres::CallbackReturnType operator() (const ceres::IterationSummary& summary) {
    //             Eigen::Matrix<double, 3, 1> eye_pos(x[0], x[1], x[2]);
    //             Sphere eye(eye_pos, eye_radius);

    //             std::vector<Circle> pupils;
    //             for (int i = 0; i < (x.size() - 3)/3; ++i) {
    //                 auto&& pupil_param_v = x.segment<3>(3 + 3 * i);
    //                 pupils.push_back(EyeModelFitter::circleFromParams(eye, PupilParams(pupil_param_v[0], pupil_param_v[1], pupil_param_v[2])));
    //             }

    //             callback(eye, pupils);

    //             return ceres::SOLVER_CONTINUE;
    //         }
    //     };
    //     options.callbacks.push_back(new CallCallbackWrapper(*this, callback, x));
    // }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << "\n";
    std::cout << "Optimized" << std::endl;

    double fit = 0;
    sphere.center = x.segment<3>(0);

     for (int i = 0; i < mSupportingPupils.size(); ++i) {
        const auto& pupil = mSupportingPupils[i];
        const Circle& unprojectedCircle = selectUnprojectedCircle(sphere, pupil.mObservationPtr->getUnprojectedCirclePair() );
        auto&& pupilParam = x.segment<3>(3 + 3 * i);
        Circle optimizedCircle = circleFromParams(sphere, PupilParams(pupilParam[0], pupilParam[1], pupilParam[2]) );
        fit += getModelSupport( unprojectedCircle , optimizedCircle );
    }

    fit /= mSupportingPupils.size();


    return fit;

}

EyeModel::Sphere EyeModel::getSphere() const {
    std::lock_guard<std::mutex> lockModel(mModelMutex);
    return mSphere;
};

EyeModel::Sphere EyeModel::getInitialSphere() const {
    std::lock_guard<std::mutex> lockModel(mModelMutex);
    return mInitialSphere;
};

double EyeModel::getMaturity() const {

    //Spatial variance
    // Our bins are just on half of the sphere and by observing different models, it turned out
    // that if a eighth of half the sphere is filled it gives a good maturity.
    // Thus we scale it that a the maturity will be 1 if a eighth is filled
    using std::floor;
    using std::pow;
    return  mSpatialBins.size()/(mTotalBins/8.0);
}

double EyeModel::getPerformance() const {
    return mPerformance;
}

double EyeModel::getFit() const {
    return mFit;
}

bool EyeModel::tryTransferNewObservations(){
    bool ownPupil = mPupilMutex.try_lock();
    if( ownPupil ){
        for( auto& pupil : mSupportingPupilsToAdd){
            mSupportingPupils.push_back( std::move(pupil) );
        }
        mSupportingPupilsToAdd.clear();
        mPupilMutex.unlock();
        mSupportingPupilSize = mSupportingPupils.size();
        return true;
    }else{
        return false;
    }

}

double EyeModel::getModelSupport(const Circle&  unprojectedCircle, const Circle& initialisedCircle) const {

    // the angle between the unprojected and the initialised circle normal tells us how good the current observation supports our current model
    // if our model is good and the camera didn't change the perspective or so, these normals should align pretty well
    const auto& n1 = unprojectedCircle.normal;
    const auto& n2 = initialisedCircle.normal;
    const double normalsAngle = n1.dot(n2);
    return normalsAngle;
}

bool EyeModel::isSpatialRelevant(const Circle& circle){

 /* In order to check if new observations are unique (not in the same area as previous one ),
     the position on the sphere (only x,y coords) are binned  (spatial binning) an inserted into the right bin.
     !! This is not a correct method to check if they are uniformly distibuted, because if x,y are uniformly distibuted
     it doesn't mean points on the spehre are uniformly distibuted.
     To uniformly distribute points on a sphere you have to check if the area is equal on the sphere of two bins.
     We just look on half of the sphere, it's like projecting a checkboard grid on the sphere, thus the bins are more dense in the projection center,
     and less dense further back.
     Still it gives good results an works for our purpose
    */
    Vector3 pupilNormal =  circle.normal; // the same as a vector from unit sphere center to the pupil center

    // calculate bin
    // values go from -1 to 1
    double x = pupilNormal.x();
    double y = pupilNormal.y();
    x = math::round(x , mBinResolution);
    y = math::round(y , mBinResolution);

    Vector2 bin(x, y);
    auto search = mSpatialBins.find(bin);

    if (search == mSpatialBins.end() || search->second == false) {

        // there is no bin at this coord or it is empty
        // so add one
        mSpatialBins.emplace(bin, true);
        double z = std::copysign(std::sqrt(1.0 - x * x - y * y),  pupilNormal.z());
        Vector3 binPositions3D(x , y, z); // for visualization
        mBinPositions.push_back(std::move(binPositions3D));
        return true;
    }

    return false;

}



const Circle& EyeModel::selectUnprojectedCircle( const Sphere& sphere,  const std::pair<const Circle, const Circle>& circles) const
{
    const Vector3& c = circles.first.center;
    const Vector3& v = circles.first.normal;
    Vector2 centerProjected = project(c, mFocalLength);
    Vector2 directionProjected = project(v + c, mFocalLength) - centerProjected;
    directionProjected.normalize();
    Vector2 eyeCenterProjected = project(sphere.center, mFocalLength);

    if ((centerProjected - eyeCenterProjected).dot(directionProjected) >= 0) {
        return circles.first;

    } else {
       return circles.second;
    }

}

void EyeModel::initialiseSingleObservation( const Sphere& sphere, Pupil& pupil) const
{
    // Ignore the circle normal, and intersect the circle
    // center projection line with the sphere
    try {
        auto pupil_center_sphere_intersect = intersect(Line3(mCameraCenter, pupil.mCircle.center.normalized()),
                                             sphere);
        auto new_pupil_center = pupil_center_sphere_intersect.first;
        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = pupil.mCircle.radius / pupil.mCircle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_center.z();
        // Parametrise this new pupil position using spherical coordinates
        Vector3 center_to_pupil = new_pupil_center - sphere.center;
        double r = center_to_pupil.norm();
        pupil.mParams.theta = acos(center_to_pupil[1] / r);
        pupil.mParams.psi = atan2(center_to_pupil[2], center_to_pupil[0]);
        pupil.mParams.radius = new_pupil_radius;
        // Update pupil circle to match parameters
        pupil.mCircle = circleFromParams(sphere,  pupil.mParams );

    } catch (no_intersection_exception&) {
        // pupil.mCircle =  Circle::Null;
        // pupil.mParams = PupilParams();
        auto pupil_radius_at_1 = pupil.mCircle.radius / pupil.mCircle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * sphere.center.z();
        pupil.mParams.radius = new_pupil_radius;
        pupil.mParams.theta = acos(pupil.mCircle.normal[1] / sphere.radius);
        pupil.mParams.psi = atan2(pupil.mCircle.normal[2], pupil.mCircle.normal[0]);
        // Update pupil circle to match parameters
        pupil.mCircle = circleFromParams(sphere,  pupil.mParams );
    }


}

Circle EyeModel::getIntersectedCircle( const Sphere& sphere, const Circle& circle) const
{
    // Ignore the circle normal, and intersect the circle
    // center projection line with the sphere
    try {
        auto pupil_center_sphere_intersect = intersect(Line3(mCameraCenter, circle.center.normalized()),
                                             sphere);
        auto new_pupil_center = pupil_center_sphere_intersect.first;
        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = circle.radius / circle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_center.z();
        // Parametrise this new pupil position using spherical coordinates
        Vector3 center_to_pupil = new_pupil_center - sphere.center;
        double r = center_to_pupil.norm();
        double theta = acos(center_to_pupil[1] / r);
        double psi = atan2(center_to_pupil[2], center_to_pupil[0]);
        double radius = new_pupil_radius;
        // Update pupil circle to match parameters
        auto pupilParams = PupilParams(theta, psi, radius);
        return  circleFromParams(sphere,  pupilParams);

    } catch (no_intersection_exception&) {
        return Circle::Null;
    }

}


Circle EyeModel::circleFromParams(const Sphere& eye, const PupilParams& params) const
{
    if (params.radius == 0)
        return Circle::Null;

    Vector3 radial = math::sph2cart<double>(double(1), params.theta, params.psi);
    return Circle(eye.center + eye.radius * radial,
                  radial,
                  params.radius);
}


void EyeModel::calculatePerformance( const Circle& unprojectedCircle , const Circle& intersectedCircle){

    double support = 0.0;
    if (unprojectedCircle != Circle::Null && intersectedCircle != Circle::Null) {  // initialise failed
        support = getModelSupport(unprojectedCircle, intersectedCircle);
    }
    mModelSupports.push_back( support );
    // calculate moving average of support
    if( mModelSupports.size() <=  mFilterWindowSize){
        mPerformance = 0.0;
        for(auto& element : mModelSupports){
            mPerformance += element;
        }
        mPerformance /= mModelSupports.size();
    }else{
        // we can optimize if the wanted window size is reached
        double first = mModelSupports.front();
        mModelSupports.pop_front();
        mPerformance += support/mFilterWindowSize - first/mFilterWindowSize;
    }
    //std::cout << "current model support: " << support  << std::endl;
    //std::cout << "average model support: " << mPerformance << std::endl;

}



} // singleeyefitter
