
#include "EyeModelFitter.h"
#include "Fit/CircleOnSphereFit.h"
#include "CircleDeviationVariance3D.h"
#include "CircleEvaluation3D.h"
#include "CircleGoodness3D.h"

#include "utils.h"
#include "ImageProcessing/cvx.h"
#include "math/intersect.h"
#include "projection.h"
#include "fun.h"

#include "mathHelper.h"
#include "math/distance.h"
#include "common/constants.h"

#include <Eigen/StdVector>
#include <algorithm>
#include <queue>

namespace singleeyefitter {


EyeModelFitter::EyeModelFitter(double focalLength, Vector3 cameraCenter) :
    mFocalLength(std::move(focalLength)),
    mCameraCenter(std::move(cameraCenter)),
    mCurrentSphere(Sphere::Null), mCurrentInitialSphere(Sphere::Null),
    mNextModelID(1),
    mActiveModelPtr(new EyeModel(mNextModelID, -1, mFocalLength, mCameraCenter)),
    mLastTimeModelAdded( Clock::now() ),
    mApproximatedFramerate(30),
    mAverageFramerate(400), // windowsize is 400, let this be slow to changes to better compensate jumps
    mLastFrameTimestamp(0),
    mPupilState(7,3,0, CV_64F),
    mLogger( pupillabs::PyCppLogger("EyeModelFitter"))

{
    mNextModelID++;

    // our model for the kalman filter
    // x,y are phi and theta
    // size is the radius of the pupil

    // 1x + 0y + deltaTime*vx + 0vy + 0.5*deltaTime^2*ax + 0ay + 0size = x
    // 0x + 1y + 0vx + deltaTime*vy + 0ax + 0.5*deltaTime^2*ay + 0size = y
    // 0x + 0y + 1vx + 0vy + deltaTime*ax + 0ay + 0size= vx
    // 0x + 0y + 0vx + 1vy + 0ax + deltaTime*ay + 0size= vy
    // 0x + 0y + 0vx + 0vy + 1ax + 0ay + 0size= ax
    // 0x + 0y + 0vx + 0vy + 0ax + 1ay + 0size= ay
    // 0x + 0y + 0vx + 0vy + 0ax + 0ay + 1size  = size

    mPupilState.measurementMatrix = (cv::Mat_<double>(3, 7) <<  1, 0, 0, 0, 0, 0, 0,
                                                                0, 1, 0, 0, 0, 0, 0,
                                                                0, 0, 0, 0, 0, 0, 1);

    cv::setIdentity(mPupilState.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(mPupilState.errorCovPost, cv::Scalar::all(1));

    cv::setIdentity(mPupilState.measurementNoiseCov, cv::Scalar::all(1e-5));
    mPupilState.measurementNoiseCov.at<double>(2,2) =  0.1; // circle size has a different variance

    mPupilState.statePost.at<double>(6) = 2.0;  //initialise the size value with the average pupil radius

}


Detector3DResult EyeModelFitter::updateAndDetect(std::shared_ptr<Detector2DResult>& observation2D , const Detector3DProperties& props , bool debug)
{

    mDebug = debug;
    Detector3DResult result;
    result.confidence = observation2D->confidence; // if we don't fit we want to take the 2D confidence
    result.timestamp = observation2D->timestamp;

    float modelSensitivity = props.model_sensitivity;

    double deltaTime = observation2D->timestamp - mLastFrameTimestamp;
    if( mLastFrameTimestamp != 0.0 ){
        mApproximatedFramerate =  static_cast<int>(1.0 / (  deltaTime ));
        mAverageFramerate.addValue(mApproximatedFramerate);
    }
    mLastFrameTimestamp = observation2D->timestamp;

    int image_height = observation2D->image_height;
    int image_width = observation2D->image_width;
    int image_height_half = image_height / 2.0;
    int image_width_half = image_width / 2.0;

    Ellipse& ellipse = observation2D->ellipse;
    ellipse.center[0] -= image_width_half;
    ellipse.center[1] = image_height_half - ellipse.center[1];
    ellipse.angle = -ellipse.angle; //take y axis flip into account



    // Observation edge data are realtive to their ROI
    cv::Rect roi = observation2D->current_roi;

    //put the edges int or coordinate system
    // edges are needed for every optimisation step
    for (cv::Point& p : observation2D->final_edges) {
        p += roi.tl();
        p.x -= image_width_half;
        p.y = image_height_half - p.y;
    }

    auto observation3DPtr = std::make_shared<const Observation>(observation2D, mFocalLength);
    bool do3DSearch = false;
    // 2d observation good enough to show to models?
    if (observation2D->confidence >= 0.7) {

        // allow each model to decide by themself if the new observation supports the model or not
        auto circleAndFit = mActiveModelPtr->presentObservation(observation3DPtr, mAverageFramerate.getAverage()  );
        auto circle = circleAndFit.first;
        auto observationFit = circleAndFit.second;

        // overwrite confidence based on 3D observation
        double confidence2D = observation2D->confidence;
        // result.confidence = confidence2D * (1.0 - observationFit.confidence) + observationFit.value * observationFit.confidence;
        result.confidence = confidence2D;
        result.circle = circle;


        // only if the detected 2d pupil fits our model well we trust it to update the Kalman filter.
        if (circle != Circle::Null && observationFit.value > 0.99 ){
            predictPupilState( deltaTime );
            auto cc  = correctPupilState( circle );
            // if (mDebug) {
            //     result.predictedCircle = cc;
            // }
        }
        else {
            do3DSearch = true;
        }

        for (auto& modelPtr : mAlternativeModelsPtrs) {
             modelPtr->presentObservation(observation3DPtr, mAverageFramerate.getAverage() );
        }

    }
    else {
        do3DSearch = true;
    }

    if (do3DSearch  && mCurrentSphere != Sphere::Null) { // if it's too weak we try to find a better one in 3D

        // since the raw edges are used to find a better circle fit
        // they need to be converted in the out coordinate system
        for (cv::Point& p : observation2D->raw_edges) {
            p += roi.tl();
            p.x -= image_width_half;
            p.y = image_height_half - p.y;
        }

        // whenever we don't have a good 2D fit we use the model's state to predict the new pupil
        // and us this as as starting point for the search
        auto predictedCircle = predictPupilState( deltaTime );

        //fitCircle(observation2D->contours, props, result );
        //filterCircle(observation2D->raw_edges, props, result);
        // filterCircle2( predictedCircle , observation2D->raw_edges, props, result);
        filterCircle3( predictedCircle , observation2D->raw_edges, props, result);


        if (result.circle != Circle::Null){
           // mPupilState.measurementNoiseCov.at<double>(0,0) = 1e-4; // circle size has a different variance
           // mPupilState.measurementNoiseCov.at<double>(1,1) = 1e-4; // circle size has a different variance

            auto estimatedCircle = correctPupilState( result.circle );
           // mPupilState.measurementNoiseCov.at<double>(0,0) = 1e-5; // circle size has a different variance
           // mPupilState.measurementNoiseCov.at<double>(1,1) = 1e-5; // circle size has a different variance

            //result.circle = estimatedCircle;
            if (mDebug) {
               result.predictedCircle = estimatedCircle;
            }
        }

    }

    // error variance
    double positionError = getPupilPositionErrorVar();
    double sizeError = getPupilSizeErrorVar();

    //std::cout << "positionError: " << positionError << std::endl;
    //std::cout << "sizeError: " << sizeError << std::endl;

    // getSphere contains a mutex and blocks if optimisation is running
    // thus we just wanna call it once and save the results
    mCurrentSphere = mActiveModelPtr->getSphere();
    mCurrentInitialSphere = mActiveModelPtr->getInitialSphere();

    result.sphere = mCurrentSphere;
    // project the sphere back to 2D
    // needed to draw it in the eye window
    result.projectedSphere = project(mCurrentSphere, mFocalLength);
    // project the circle back to 2D
    // needed for some calculations in 2D later (calibration)
    if(result.circle != Circle::Null){
        result.ellipse  = Ellipse(project(result.circle,mFocalLength));
    }
    else{
        result.confidence = 0.0;
        result.ellipse = Ellipse::Null;
   }

    // contains the logic for building alternative models if the current one is bad
    checkModels(modelSensitivity,observation2D->timestamp );
    result.modelID = mActiveModelPtr->getModelID();
    result.modelBirthTimestamp = mActiveModelPtr->getBirthTimestamp();
    result.modelConfidence = mActiveModelPtr->getConfidence();

    if (mDebug) {
        result.models.reserve(mAlternativeModelsPtrs.size() + 1);


        ModelDebugProperties props;
        props.sphere = mActiveModelPtr->getSphere();
        props.initialSphere = mActiveModelPtr->getInitialSphere();
        props.binPositions = mActiveModelPtr->getBinPositions();
        props.maturity = mActiveModelPtr->getMaturity();
        props.solverFit = mActiveModelPtr->getSolverFit();
        props.confidence = mActiveModelPtr->getConfidence();
        props.performance = mActiveModelPtr->getPerformance();
        props.performanceGradient = mActiveModelPtr->getPerformanceGradient();
        props.modelID = mActiveModelPtr->getModelID();
        props.birthTimestamp = mActiveModelPtr->getBirthTimestamp();
        result.models.push_back(std::move(props));

        for (const auto& modelPtr : mAlternativeModelsPtrs) {

            ModelDebugProperties props;
            props.sphere = modelPtr->getSphere();
            props.initialSphere = modelPtr->getInitialSphere();
            props.binPositions = modelPtr->getBinPositions();
            props.maturity = modelPtr->getMaturity();
            props.solverFit = modelPtr->getSolverFit();
            props.confidence = modelPtr->getConfidence();
            props.performance = modelPtr->getPerformance();
            props.performanceGradient = modelPtr->getPerformanceGradient();
            props.modelID = modelPtr->getModelID();
            props.birthTimestamp = modelPtr->getBirthTimestamp();
            result.models.push_back(std::move(props));

        }
    }

    return result;

}


void EyeModelFitter::checkModels( float sensitivity,double frame_timestamp )
{

    using namespace std::chrono;

    static const int maxAltAmountModels  = 3;
    static const double minMaturity  = 0.15;
    const double minPerformance = sensitivity;
    static const seconds altModelExpirationTime(10);
    static const seconds minNewModelTime(3);
    static const double gradientChangeThreshold = -2.0e-05; // with this we are also sensitive to changes even if the performance is still above the threshold

    Clock::time_point  now( Clock::now() );

    /* whenever our current model's performance is below the threshold or the performance decreases rapidly (performance gradient)
       we try to create an alternative model
    */
    //std::cout << "current performance gradient: " << mActiveModelPtr->getPerformanceGradient() << std::endl;
    if(  mActiveModelPtr->getPerformance() < minPerformance || mActiveModelPtr->getPerformanceGradient() <= gradientChangeThreshold){

        mLastTimePerformancePenalty = now;
        auto lastTimeAdded =  duration_cast<seconds>(now - mLastTimeModelAdded);

        if( mAlternativeModelsPtrs.size() <= maxAltAmountModels &&
            mActiveModelPtr->getMaturity() > minMaturity &&
            lastTimeAdded  > minNewModelTime )
        {
            mAlternativeModelsPtrs.emplace_back(  new EyeModel(mNextModelID , frame_timestamp, mFocalLength, mCameraCenter ) );
            mNextModelID++;
            mLastTimeModelAdded = now;
        }

    }else if( mActiveModelPtr->getPerformance() > minPerformance /*&& mActiveModelPtr->getPerformanceGradient() >  0.0*/ ) {
        // kill other models whenever the performance is good enough AND the performance doesn't decrease
        mAlternativeModelsPtrs.clear();
        mLastTimeModelAdded = now - minNewModelTime - seconds(1); // so we can add a new model right away
    }

    if(mAlternativeModelsPtrs.size() == 0)
        return; // early exit

    // sort the model with increasing fit
    const auto sortFit = [](const EyeModelPtr&  a , const EyeModelPtr& b){
        return a->getSolverFit() < b->getSolverFit();
    };
    mAlternativeModelsPtrs.sort(sortFit);

    bool foundNew = false;
    // now get the first alternative model where the performance and maturity is higher then the current one
    for( auto& modelptr : mAlternativeModelsPtrs){

        if(modelptr->getMaturity() > minMaturity &&  mActiveModelPtr->getPerformance() < modelptr->getPerformance() ){
            mActiveModelPtr.reset( modelptr.release() );
            mAlternativeModelsPtrs.clear(); // we got a better one, let's remove others
            foundNew = true;
            break;
        }
    }

    auto lastPenalty =  duration_cast<seconds>(now - mLastTimePerformancePenalty);
    // if we didn't find a better one after repeatedly looking, remove all of them and start new
    if( !foundNew && lastPenalty > altModelExpirationTime ){

        mAlternativeModelsPtrs.clear();
        mActiveModelPtr.reset(  new EyeModel(mNextModelID , frame_timestamp, mFocalLength, mCameraCenter ));
        mNextModelID++;
    }


}

void EyeModelFitter::reset()
{
    mNextModelID = 1;
    mAlternativeModelsPtrs.clear();
    mActiveModelPtr = EyeModelPtr( new EyeModel(mNextModelID , -1, mFocalLength, mCameraCenter ));
    mLastTimeModelAdded =  Clock::now();
    mCurrentSphere = Sphere::Null;
    mCurrentInitialSphere = Sphere::Null;
    //mLogger.setLogLevel( pupillabs::PyCppLogger::LogLevel::DEBUG);
    //mLogger.info("Reset models");
    //mLogger.error("Reset models");

}

// void  EyeModelFitter::fitCircle(const Contours_2D& contours2D , const Detector3DProperties& props,  Detector3DResult& result) const
// {

//     if (contours2D.size() == 0)
//         return;

//     Contours3D contoursOnSphere  = unprojectContours( contours2D );


//     double minRadius = props.pupil_radius_min;
//     double maxRadius =  props.pupil_radius_max;

//     if (mPreviousPupil.radius != 0.0) {

//         minRadius = std::max(mPreviousPupil.radius * 0.85, minRadius );
//         maxRadius = std::min(mPreviousPupil.radius * 1.25, maxRadius );
//     }
//     const double maxDiameter = maxRadius * 2.0;

//     //final_candidate_contours.clear(); // otherwise we fill this infinitly

//     //first we want to filter out the bad stuff, too short ones
//     const auto contour_size_min_pred = [](const std::vector<Vector3>& contour) {
//         return contour.size() >= 3;
//     };
//     contoursOnSphere = singleeyefitter::fun::filter(contour_size_min_pred , contoursOnSphere);

//     if (contoursOnSphere.size() == 0)
//         return ;

//     // sort the contours so the contour with the most points is at the begining
//     std::sort(contoursOnSphere.begin(), contoursOnSphere.end(), [](const std::vector<Vector3>& a, const std::vector<Vector3>& b) { return a.size() < b.size();});

//     // saves the best solution and just the Vector3Ds not every single contour
//     Contours3D bestSolution;
//     Circle bestCircle;
//     double bestVariance = std::numeric_limits<double>::infinity();
//     double bestGoodness = 0.0;
//     double bestResidual = 0.0;

//     auto circleFitter = CircleOnSphereFitter<double>(mCurrentSphere);
//     auto circleEvaluation = CircleEvaluation3D<double>(mCameraCenter, mCurrentSphere, props.max_fit_residual, minRadius, maxRadius);
//     auto circleVariance = CircleDeviationVariance3D<double>();
//     auto circleGoodness = CircleGoodness3D<double>();


//     auto pruning_quick_combine = [&](const Contours3D & contours,  int max_evals = 1e20, int max_depth = 5) {
//         // describes different combinations of contours
//         typedef std::set<int> Path;
//         // combinations we wanna test
//         std::queue<Path> unvisited;

//         // contains all the indices for the contours, which altogther fit best
//         std::vector<Path> results;

//         // contains bad paths, we won't test again
//         // even a superset is not tested again, because if a subset is bad, we can't make it better if more contours are added
//         std::vector<Path> prune;
//         prune.reserve(std::pow(contours.size() , 3));   // we gonna prune a lot if we have alot contours
//         int eval_count = 0;
//         //std::cout << "size:" <<  contours.size()  << std::endl;
//         //std::cout << "possible combinations: " <<  std::pow(2,contours.size()) + 1<< std::endl;

//         // contains the first moment of each contour
//         // we precalculate this inorder to prune contours combinations if the distance of these are to long
//         std::vector<Vector3> moments;
//         moments.reserve(contours.size());

//         // enqueue all contours as starting point
//         // and calculate moment
//         for (int i = 0; i < contours.size(); i++) {
//             unvisited.emplace(std::initializer_list<int> {i});

//             Vector3 m = std::accumulate(contours[i].begin(), contours[i].end(), Vector3(0, 0, 0), std::plus<Vector3>());
//             m /= contours[i].size();
//             moments.push_back(m);
//         }

//         // inorder to minimize the search space we already prune combinations, which can't fit ,before the search starts
//         int prune_count = 0;

//         for (int i = 0; i < contours.size(); i++) {
//             auto& a = moments[i];

//             for (int j = i + 1; j < contours.size(); j++) {
//                 auto& b = moments[j];
//                 double distance_squared  = (a - b).squaredNorm();

//                 if (distance_squared >  std::pow(maxDiameter * 1.5, 2.0)) {
//                     prune.emplace_back(std::initializer_list<int> {i, j});
//                     prune_count++;
//                 }
//             }
//         }

//         // std::cout << "pruned " << prune_count << std::endl;

//         while (!unvisited.empty() && eval_count <= max_evals) {
//             eval_count++;
//             //take a path and combine it with others to see if the fit gets better
//             Path current_path = unvisited.front();
//             unvisited.pop();

//             if (current_path.size() <= max_depth) {
//                 bool includes_bad_paths = fun::isSubset(current_path, prune);

//                 if (!includes_bad_paths) {
//                     int size = 0;

//                     for (int j : current_path) { size += contours.at(j).size(); };

//                     Contour3D test_contour;

//                     Contours3D test_contours;

//                     test_contour.reserve(size);

//                     std::set<int> test_contour_indices;

//                     //concatenate contours to one contour
//                     for (int k : current_path) {
//                         const Contour3D& c = contours.at(k);
//                         test_contours.push_back(c);
//                         test_contour.insert(test_contour.end(), c.begin(), c.end());
//                         test_contour_indices.insert(k);
//                     }

//                     //we have not tested this and a subset of this was sucessfull before

//                     // need at least 3 points
//                     if (!circleFitter.fit(test_contour)) {
//                         std::cout << "Error! Too little points!" << std::endl; // filter too short ones before
//                     }

//                     // we got a circle fit
//                     Circle current_circle = circleFitter.getCircle();
//                     // see if it's even a candidate
//                     double variance =  circleVariance(current_circle , test_contours);

//                     if (variance <  props.max_circle_variance) {
//                         //yes this was good, keep as solution
//                         //results.push_back(test_contour_indices);

//                         //lets explore more by creating paths to each remaining node
//                         for (int l = (*current_path.rbegin()) + 1 ; l < contours.size(); l++) {
//                             // if a new contour is to far away from the current circle center, we can also ignore it
//                             // Vector3 contour_moment = moments.at(l);
//                             // double distance_squared = (current_circle.center - contour_moment).squaredNorm();
//                             // if( distance_squared <   std::pow(pupil_max_radius * 1.5, 2.0) ){
//                             //     unvisited.push(current_path);
//                             //     unvisited.back().insert(l); // add a new path
//                             // }
//                             unvisited.push(current_path);
//                             unvisited.back().insert(l); // add a new path
//                         }

//                         double residual = circleFitter.calculateResidual(test_contour);
//                         bool isCandidate = circleEvaluation(current_circle, residual);
//                         double goodness =  circleGoodness(current_circle , test_contours);

//                         // if (isCandidate)
//                         //     final_candidate_contours.push_back(test_contours);

//                         //check if this one is better then the best one and swap
//                         if (isCandidate &&  goodness > bestGoodness) {

//                             bestResidual = residual;
//                             bestVariance = variance;
//                             bestGoodness = goodness;
//                             bestCircle = current_circle;
//                             bestSolution = test_contours;
//                         }

//                     } else {
//                         prune.push_back(current_path);
//                     }
//                 }
//             }
//         }

//         //std::cout << "tried: "  << eval_count  << std::endl;
//         //return results;
//     };

//     pruning_quick_combine(contoursOnSphere, props.combine_evaluation_max, props.combine_depth_max);

//     //std::cout << "residual: " <<  bestResidual << std::endl;
//     //std::cout << "goodness: " <<  bestGoodness << std::endl;
//     //std::cout << "variance: " <<  bestVariance << std::endl;
//     result.circle = std::move(bestCircle);
//     result.fittedCircleContours = std::move(bestSolution); // save this for debuging
//     result.fitGoodness = bestGoodness;
//     result.contours = std::move( contoursOnSphere );
//     // project the circle back to 2D
//     // need for some calculations in 2D later (calibration)
//     result.ellipse = Ellipse(project(bestCircle, mFocalLength));

// }

// Contours3D EyeModelFitter::unprojectContours(const Contours_2D& contours) const
// {
//     Contours3D contoursOnSphere;
//     contoursOnSphere.resize(contours.size());
//     int i = 0;
//     //TODO handle contours with no intersection points, because they get closed
//     for (auto& contour : contours) {
//         for (auto& point : contour) {
//             Vector3 point3D(point.x, point.y , mFocalLength);
//             Vector3 direction = point3D - mCameraCenter;

//             try {
//                 // we use the eye properties of the current eye, when ever we call this
//                 const auto& unprojectedPoint = intersect(Line3(mCameraCenter,  direction.normalized()), mCurrentSphere);
//                 contoursOnSphere[i].push_back(std::move(unprojectedPoint.first));

//             } catch (no_intersection_exception&) {
//                 // if there is no intersection we don't do anything
//             }
//         }
//         i++;
//     }
//     return contoursOnSphere;

// }

Edges3D EyeModelFitter::unprojectEdges(const Edges2D& edges) const
{
    Edges3D edgesOnSphere;
    edgesOnSphere.reserve(edges.size());

    for (auto& edge : edges) {
        Vector3 point3D(edge.x, edge.y , mFocalLength);
        Vector3 direction = point3D - mCameraCenter;

        std::pair<Vector3, Vector3> unprojectedPoints;
        // we use the eye properties of the current eye, when ever we call this
        bool didIntersect = intersect(Line3(mCameraCenter,  direction.normalized()), mCurrentSphere, unprojectedPoints);

        if( didIntersect)
            edgesOnSphere.push_back(std::move(unprojectedPoints.first));

    }

    return edgesOnSphere;

}

// void  EyeModelFitter::filterCircle(const Edges2D& rawEdges , const Detector3DProperties& props,  Detector3DResult& result) const
// {

//     if (rawEdges.size() == 0 || mPreviousPupil == Circle::Null)
//         return;


//     Edges3D edgesOnSphere = unprojectEdges(rawEdges);

//     //Inorder to filter the edges depending on the distance of the previous pupil center
//     // imagine a sphere with center equal to the previous pupil center (pupilcenters are always on the sphere )
//     // and sphere radius equal the distance from sphere center to pupil border
//     double h =  mCurrentSphere.radius - std::sqrt(mCurrentSphere.radius * mCurrentSphere.radius - mPreviousPupil.radius * mPreviousPupil.radius);
//     double pupilSphereRadiusSquared =  2.0 * mCurrentSphere.radius  * h;
//     double pupilSphereRadius = std::sqrt(pupilSphereRadiusSquared);
//     Vector3 pupilSphereCenter = mPreviousPupil.center;

//     const double delta = std::pow(1.5, 2);
//     const double maxFilterDistanceSquared = pupilSphereRadiusSquared * delta;
//     auto regionFilter = [&](const Vector3 & point) {
//         double distanceSquared = (point - pupilSphereCenter).squaredNorm();
//         return  distanceSquared < maxFilterDistanceSquared;
//     };

//     auto filteredEdges = fun::filter(regionFilter, edgesOnSphere);

//     // now we got all edges in the surrounding of the previous pupil
//     // let find the circle where most edges support the circle including a certain region around the circle border

//     const double maxAngularVelocity = 0.1;  //TODO this should depend on the realy velocity calculated per frame

//     Vector3 c  = mPreviousPupil.center - mCurrentSphere.center;
//     // search space is in spehrical coordinates
//     Vector2 previousPupilCenter  = math::cart2sph(c);
//     const double maxTheta = previousPupilCenter.x() + maxAngularVelocity ;
//     const double maxPsi = previousPupilCenter.y() + maxAngularVelocity ;
//     const double minTheta = previousPupilCenter.x() - maxAngularVelocity;
//     const double minPsi = previousPupilCenter.y() - maxAngularVelocity;

//     const double stepSizeAngle = 0.01; // in radian

//     // defined in pixel space and recalculated for 3D space further down
//     const int bandWidthPixel =  4 ;

//     int maxEdgeCount = 0;
//     Vector3 bestCircleCenter(0, 0, 0);
//     double bestDistanceVariance = std::numeric_limits<double>::max();
//     Edges3D inliers;
//     Edges3D finalInliers;

//     for (double i = minTheta; i <= maxTheta; i += stepSizeAngle) {
//         for (double j = minPsi; j <=  maxPsi; j += stepSizeAngle) {

//             // from here in cartesian again
//             // if we use cartesian we can just compare the distances from the pupil sphere center
//             // all this happens in world coordinates
//             const Vector3 newPupilCenter  = mCurrentSphere.center + math::sph2cart(mCurrentSphere.radius, i, j);

//             const double bandWidth =  bandWidthPixel * newPupilCenter.z() / mFocalLength ;
//             const double bandWidthHalf = bandWidth / 2.0 ;
//             const double maxDistanceSquared  = std::pow(pupilSphereRadius + bandWidthHalf, 2) ;
//             const double minDistanceSquared  = std::pow(pupilSphereRadius - bandWidthHalf, 2) ;

//             if(mDebug)
//                 inliers.clear();

//             int  edgeCount = 0;
//             //count all edges which fall into this current circle
//             for (const auto& e : filteredEdges) {

//                 double distanceSquared = (e - newPupilCenter).squaredNorm();
//                 if (distanceSquared < maxDistanceSquared && distanceSquared > minDistanceSquared) {
//                     edgeCount++;
//                     if(mDebug)
//                         inliers.push_back(e);
//                 }
//             }

//             if (edgeCount > maxEdgeCount  ) {
//                 bestCircleCenter = newPupilCenter;
//                 maxEdgeCount = edgeCount;
//                 if(mDebug)
//                     finalInliers = std::move(inliers);
//             }

//         }
//     }

//     if (maxEdgeCount != 0) {
//         result.circle.center = bestCircleCenter;
//         result.circle.normal = (bestCircleCenter - mCurrentSphere.center).normalized() ;
//         result.circle.radius = mPreviousPupil.radius;

//         // project the circle back to 2D
//         // needed for some calculations in 2D later (calibration)
//         result.ellipse  = Ellipse(project(result.circle,mFocalLength));

//         double circumference = result.ellipse.circumference();
//         result.confidence =  std::min(maxEdgeCount / circumference, 1.0 ) ;
//     }


//     if( mDebug )
//       result.edges = std::move(finalInliers);  // visualize

// }

void  EyeModelFitter::filterCircle2( const Circle& predictedCircle, const Edges2D& rawEdges , const Detector3DProperties& props,  Detector3DResult& result) const
{

    if (rawEdges.size() == 0 )
        return;


    Edges3D edgesOnSphere = unprojectEdges(rawEdges);

    //Inorder to filter the edges depending on the distance of the predicted pupil center
    // imagine a sphere with center equal to the predicted pupil center (pupilcenters are always on the sphere )
    // and sphere radius equal the distance of the predicted pupil radius
    double h =  mCurrentSphere.radius - std::sqrt(mCurrentSphere.radius * mCurrentSphere.radius - predictedCircle.radius * predictedCircle.radius);
    double pupilSphereRadiusSquared =  2.0 * mCurrentSphere.radius  * h;
    double pupilSphereRadius = std::sqrt(pupilSphereRadiusSquared);
    Vector3 pupilSphereCenter = predictedCircle.center;

    const double delta = std::pow(1.5, 2);
    const double maxFilterDistanceSquared = pupilSphereRadiusSquared * delta;
    auto regionFilter = [&](const Vector3 & point) {
        double distanceSquared = (point - pupilSphereCenter).squaredNorm();
        return  distanceSquared < maxFilterDistanceSquared;
    };

    auto filteredEdges = fun::filter(regionFilter, edgesOnSphere);

    // now we got all edges in the surrounding of the predicted pupil
    // let's find the circle where most edges support the circle including a certain region around the circle border

    const double maxAngularVelocity = 0.06;  //TODO this should depend on the realy velocity calculated per frame

    Vector3 c  = predictedCircle.center - mCurrentSphere.center;
    // search space is in spehrical coordinates
    Vector2 predictedPupilCenter  = math::cart2sph(c);
    const double maxTheta = predictedPupilCenter.x() + maxAngularVelocity ;
    const double maxPsi = predictedPupilCenter.y() + maxAngularVelocity ;
    const double minTheta = predictedPupilCenter.x() - maxAngularVelocity;
    const double minPsi = predictedPupilCenter.y() - maxAngularVelocity;

    const double stepSizeAngle = 0.005; // in radian

    // defined in pixel space and recalculated for 3D space further down
    const int bandWidthPixel =  4 ;

    int maxEdgeCount = 0;
    Vector3 bestCircleCenter(0, 0, 0);
    double bestCircleRadius = 0.0;
    double bestDistanceVariance = std::numeric_limits<double>::max();
    Edges3D inliers;
    Edges3D finalInliers;

    for (double i = minTheta; i <= maxTheta; i += stepSizeAngle) {
        for (double j = minPsi; j <=  maxPsi; j += stepSizeAngle) {

            // from here in cartesian again
            // if we use cartesian we can just compare the distances from the pupil sphere center
            // all this happens in world coordinates
            const Vector3 newPupilCenter  = mCurrentSphere.center + math::sph2cart(mCurrentSphere.radius, i, j);

            const double bandWidth =  bandWidthPixel * newPupilCenter.z() / mFocalLength ;
            const double bandWidthHalf = bandWidth / 2.0 ;
            const double maxDistanceSquared  = std::pow(pupilSphereRadius + bandWidthHalf, 2) ;
            const double minDistanceSquared  = std::pow(pupilSphereRadius - bandWidthHalf, 2) ;

            if(mDebug)
                inliers.clear();

            int  edgeCount = 0;
            double accRadius = 0.0;
            //count all edges which fall into this current circle
            for (const auto& e : filteredEdges) {

                double distanceSquared = (e - newPupilCenter).squaredNorm();
                if (distanceSquared < maxDistanceSquared && distanceSquared > minDistanceSquared) {
                    edgeCount++;
                    accRadius += std::sqrt(distanceSquared);
                    if(mDebug)
                        inliers.push_back(e);
                }
            }

            if (edgeCount > maxEdgeCount  ) {
                bestCircleCenter = newPupilCenter;
                bestCircleRadius = accRadius / edgeCount;
                maxEdgeCount = edgeCount;
                if(mDebug)
                    finalInliers = std::move(inliers);
            }

        }
    }

    if (maxEdgeCount != 0) {
        result.circle.center = bestCircleCenter;
        result.circle.normal = (bestCircleCenter - mCurrentSphere.center).normalized() ;
        result.circle.radius = predictedCircle.radius;
        result.circle.radius = bestCircleRadius;

        // project the circle back to 2D
        // needed for some calculations in 2D later (calibration)
        result.ellipse  = Ellipse(project(result.circle,mFocalLength));

        double circumference = result.ellipse.circumference();
        result.confidence =  std::min(maxEdgeCount / circumference, 1.0 ) ;
    }


    if( mDebug )
      result.edges = std::move(finalInliers);  // visualize
}

void  EyeModelFitter::filterCircle3( const Circle& predictedCircle, const Edges2D& rawEdges , const Detector3DProperties& props,  Detector3DResult& result) const
{

    if (rawEdges.size() == 0 )
        return;


    Edges3D edgesOnSphere = unprojectEdges(rawEdges);
    Edges3D filteredEdges;
    Edges3D finalInliers;
    int maxEdgeCount;
    auto searchCenter  = [this, &finalInliers, &edgesOnSphere, &maxEdgeCount, &predictedCircle, &filteredEdges ]( Vector3 searchCenter,   double positionVariance = 0.06, double searchStep = 0.005, int bandWidthPixel = 4 ) -> Circle {

        //Inorder to filter the edges depending on the distance of the predicted pupil center
        // imagine a sphere with center equal to the predicted pupil center (pupilcenters are always on the sphere )
        // and sphere radius equal the distance of the predicted pupil radius
        double h =  mCurrentSphere.radius - std::sqrt(mCurrentSphere.radius * mCurrentSphere.radius - predictedCircle.radius * predictedCircle.radius);
        double pupilSphereRadiusSquared =  2.0 * mCurrentSphere.radius  * h;
        double pupilSphereRadius = std::sqrt(pupilSphereRadiusSquared);
        Vector3 pupilSphereCenter = predictedCircle.center;

        // delta depends on the postionvariance so the window has the right size
        // position variance is in radians
        const double delta = std::sin(positionVariance) * mCurrentSphere.radius;
        const double deltaSquared = std::pow(delta, 2);
        const double maxFilterDistanceSquared = pupilSphereRadiusSquared + deltaSquared;
        auto regionFilter = [&](const Vector3 & point) {
            double distanceSquared = (point - pupilSphereCenter).squaredNorm();
            return  distanceSquared < maxFilterDistanceSquared;
        };

        filteredEdges = fun::filter(regionFilter, edgesOnSphere);

        // now we got all edges in the surrounding of the predicted pupil
        // let's find the circle where most edges support the circle including a certain region around the circle border


        const double maxAngularVelocity = positionVariance;  //TODO this should depend on the realy velocity calculated per frame

        Vector3 c  = searchCenter - mCurrentSphere.center;
        // search space is in spehrical coordinates
        Vector2 predictedPupilCenter  = math::cart2sph(c);
        const double maxTheta = predictedPupilCenter.x() + maxAngularVelocity ;
        const double maxPsi = predictedPupilCenter.y() + maxAngularVelocity ;
        const double minTheta = predictedPupilCenter.x() - maxAngularVelocity;
        const double minPsi = predictedPupilCenter.y() - maxAngularVelocity;

        const double stepSizeAngle = searchStep; // in radian

        // defined in pixel space and recalculated for 3D space further down
        //const int bandWidthPixel =  4 ;

        maxEdgeCount = 0;
        Vector3 bestCircleCenter(0, 0, 0);
        double bestCircleRadius = 0.0;
        double bestDistanceVariance = std::numeric_limits<double>::max();
        Edges3D inliers;

        for (double i = minTheta; i <= maxTheta; i += stepSizeAngle) {
            for (double j = minPsi; j <=  maxPsi; j += stepSizeAngle) {

                // from here in cartesian again
                // if we use cartesian we can just compare the distances from the pupil sphere center
                // all this happens in world coordinates
                const Vector3 newPupilCenter  = mCurrentSphere.center + math::sph2cart(mCurrentSphere.radius, i, j);

                const double bandWidth =  bandWidthPixel * newPupilCenter.z() / mFocalLength ;
                const double bandWidthHalf = bandWidth / 2.0 ;
                const double maxDistanceSquared  = std::pow(pupilSphereRadius + bandWidthHalf, 2) ;
                const double minDistanceSquared  = std::pow(pupilSphereRadius - bandWidthHalf, 2) ;

                if(mDebug)
                    inliers.clear();

                int  edgeCount = 0;
                double accRadius = 0.0;
                //count all edges which fall into this current circle
                for (const auto& e : filteredEdges) {

                    double distanceSquared = (e - newPupilCenter).squaredNorm();
                    if (distanceSquared < maxDistanceSquared && distanceSquared > minDistanceSquared) {
                        edgeCount++;
                        accRadius += std::sqrt(distanceSquared);
                        if(mDebug)
                            inliers.push_back(e);
                    }
                }

                if (edgeCount > maxEdgeCount  ) {
                    bestCircleCenter = newPupilCenter;
                    bestCircleRadius = accRadius / edgeCount;
                    maxEdgeCount = edgeCount;
                    if(mDebug)
                        finalInliers = std::move(inliers);
                }

            }
        }
        if (maxEdgeCount != 0){
            Circle circle;
            circle.center = bestCircleCenter;
            circle.normal = (bestCircleCenter - mCurrentSphere.center).normalized() ;
            circle.radius = bestCircleRadius;
            return circle;
        }else{
            return Circle::Null;
        }


    };

    double positionVariance = 0.2;
    double searchStep = 0.05;
    int bandWidthPixel = 6;

    auto circle = searchCenter(predictedCircle.center, positionVariance, searchStep, bandWidthPixel);

    positionVariance = 0.05;
    searchStep = 0.01;
    bandWidthPixel = 2;

    circle  = searchCenter(circle.center, positionVariance, searchStep, bandWidthPixel);

    if (circle != Circle::Null ) {

        result.circle = circle;
        // project the circle back to 2D
        // needed for some calculations in 2D later (calibration)
        result.ellipse  = Ellipse(project(result.circle,mFocalLength));

        double circumference = result.ellipse.circumference();
        result.confidence =  std::min(maxEdgeCount / circumference, 1.0 ) ;
    }


    if( mDebug )
      result.edges = std::move(filteredEdges);  // visualize
}


Circle EyeModelFitter::predictPupilState( double deltaTime ){


    // correlates position and velocity
    // x,y are phi and theta
    // 1x + 0y + deltaTime*vx + 0vy + 0.5*deltaTime^2*ax + 0ay  = x
    // 0x + 1y + 0vx + deltaTime*vy + 0ax + 0.5*deltaTime^2*ay  = y
    // 0x + 0y + 1vx + 0vy + deltaTime*ax + 0ay = vx
    // 0x + 0y + 0vx + 1vy + 0ax + deltaTime*ay = vy
    // 0x + 0y + 0vx + 0vy + 1ax + 0ay = ax
    // 0x + 0y + 0vx + 0vy + 0ax + 1ay = ay
   mPupilState.transitionMatrix = (cv::Mat_<double>(7, 7) << 1, 0, deltaTime, 0, 0.5*deltaTime*deltaTime , 0, 0,
                                                          0, 1, 0, deltaTime, 0 , 0.5*deltaTime*deltaTime, 0,
                                                          0, 0, 1, 0, deltaTime, 0,0,
                                                          0, 0, 0, 1, 0, deltaTime,0,
                                                          0, 0, 0, 0, 1, 0,0,
                                                          0, 0, 0, 0, 0, 1,0,
                                                          0, 0, 0, 0, 0, 0,1);

    cv::Mat pupilStatePrediction = mPupilState.predict();
    double theta = pupilStatePrediction.at<double>(0);
    double psi = pupilStatePrediction.at<double>(1);
    double radius = pupilStatePrediction.at<double>(6);
    return circleOnSphere( mCurrentSphere, theta, psi, radius );

}


Circle EyeModelFitter::correctPupilState( const Circle& circle){

    Vector2 params = paramsOnSphere(mCurrentSphere, circle);
    cv::Mat meausurement = (cv::Mat_<double>(3,1) << params[0], params[1], circle.radius );
    auto estimated = mPupilState.correct( meausurement );

    double theta = estimated.at<double>(0);
    double psi = estimated.at<double>(1);
    double radius = estimated.at<double>(6);
    auto estimatedCircle = circleOnSphere( mCurrentSphere , theta, psi, radius );
    return estimatedCircle;
}

double EyeModelFitter::getPupilPositionErrorVar() const {

    // error variance
    double thetaError = mPupilState.errorCovPost.at<double>(0,0);
    double psiError = mPupilState.errorCovPost.at<double>(1,1);
   // std::cout << "te: " << thetaError << std::endl;
   // std::cout << "pE: " << psiError << std::endl;
    // for now let's just use the average from both values
    return (thetaError + psiError) * 0.5;

}

double EyeModelFitter::getPupilSizeErrorVar() const {

    // error variance
    double sizeError = mPupilState.errorCovPost.at<double>(6,6);
    return sizeError;

}

} // singleeyefitter
