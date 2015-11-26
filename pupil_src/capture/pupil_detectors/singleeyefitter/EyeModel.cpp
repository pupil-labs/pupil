

#include "EyeModel.h"

#include <algorithm>

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



EyeModel::EyeModel(EyeModel&& that)
{

    {
        std::lock_guard<std::mutex> lock(that.mObservationMutex);
        mObservations = std::move(that.mObservations);
    }
    {
        std::lock_guard<std::mutex> lock(that.mSphereMutex);
        mSphere = std::move(that.mSphere);
    }
    mSpatialBins = std::move(that.mSpatialBins);
    mBinPositions = std::move(that.mBinPositions);
    mWorker = std::move(that.mWorker);
    mAverageModelSupport = std::move(that.mAverageModelSupport);
    mPerformance = std::move(that.mPerformance);
    mMaturity = std::move(that.mMaturity);
    mFocalLength = std::move(that.mFocalLength);
}

EyeModel::~EyeModel(){


    // wait for thread ?
    //TODO reset when thread is running
    // see what happen :)
}

EyeModel::CircleParams::CircleParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius)
{
}

EyeModel::CircleParams::CircleParams() : theta(0), psi(0), radius(0)
{
}



Circle EyeModel::presentObservation(const ObservationPtr observation)
{


    Circle circle = Circle::Null;
    bool should_add_observation = false;

    //Check for properties if it's a candidate we can use
    if (mSphere != Sphere::Null) {

        // select the right circle depending on the current model
        const Circle& unprojectedCircle = selectUnprojectedCircle(observation->mUnprojectedCirclePair);
        // initialised circle. circle parameters addapted to our current eye model
        circle = getIntersectedCircle(unprojectedCircle).circle;

        if (unprojectedCircle != Circle::Null && circle != Circle::Null) {  // initialise failed

            double support = getModelSupport(unprojectedCircle, circle);

            //std::cout << "support: " << support  << std::endl;
            if (support > 0.97) {

                if (isSpatialRelevant(circle)) {
                    should_add_observation = true;
                } else {
                    //std::cout << " spatial check failed"  << std::endl;
                }

            } else {
                std::cout << "doesn't support current model "  << std::endl;
            }


        } else {
            std::cout << "no valid circles"  << std::endl;
        }

    } else { // no valid sphere yet
        std::cout << "add without check" << std::endl;
        should_add_observation = true;
    }


    if (should_add_observation) {
        //std::cout << "add" << std::endl;

        //if the observation passed all tests we can add it
        mNewObservations.push_back(std::move(observation));

        std::cout << "new observation size " << mNewObservations.size() << std::endl;

        //refine model every 50 new pupils
        transferNewObservations();
        auto circles = findSphereCenter();
        circles = initialiseModel(circles);
        refineWithEdges(circles);

        if (mSphere != Sphere::Null){
            // if we have change the sphere, get the new circle
            const Circle& unprojectedCircle = selectUnprojectedCircle(observation->mUnprojectedCirclePair);
            circle = getIntersectedCircle(unprojectedCircle).circle;
        }


    }

    return circle;
}

std::vector<EyeModel::CircleDescription> EyeModel::findSphereCenter( bool use_ransac /*= true*/)
{
    using math::sq;
    std::lock_guard<std::mutex> lockObservations(mObservationMutex);
    std::vector<EyeModel::CircleDescription> pupils;

    if (mObservations.size() < 2) {
        return pupils;
    }
    std::cout << "find s center" << std::endl;
    const double pupil_radius = 1;
    const double eye_z = 57;
    //std::vector<std::pair<Circle, Circle>> pupil_unprojection_pairs;
    std::vector<Line> pupil_gazelines_proj;
    // TODO do this just once
    for (const auto observationPtr : mObservations) {
        // Get pupil circles (up to depth)
        //
        // Do a per-image unprojection of the pupil ellipse into the two fixed
        // size circles that would project onto it. The size of the circles
        // doesn't matter here, only their center and normal does.
        auto unprojection_pair = observationPtr->mUnprojectedCirclePair;
        // Get projected circles and gaze vectors
        //
        // Project the circle centers and gaze vectors down back onto the image
        // plane. We're only using them as line parametrisations, so it doesn't
        // matter which of the two centers/gaze vectors we use, as the
        // two gazes are parallel and the centers are co-linear.
        const auto& c = unprojection_pair.first.center;
        const auto& v = unprojection_pair.first.normal;
        Vector2 c_proj = project(c, mFocalLength);
        Vector2 v_proj = project(v + c, mFocalLength) - c_proj;
        v_proj.normalize();
        //pupil_unprojection_pairs.push_back(std::move(unprojection_pair));
        pupil_gazelines_proj.emplace_back(c_proj, v_proj);
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
    Vector2 eye_center_proj;
    bool valid_eye;

    if (/*use_ransac*/ false) {
        // auto indices = fun::range_<std::vector<size_t>>(pupil_gazelines_proj.size());
        // const int n = 2;
        // double w = 0.3;
        // double p = 0.9999;
        // int k = ceil(log(1 - p) / log(1 - pow(w, n)));
        // double epsilon = 10;
        // auto huber_error = [&](const Vector2 & point, const Line & line) {
        //     double dist = euclidean_distance(point, line);

        //     if (sq(dist) < sq(epsilon))
        //         return sq(dist) / 2;
        //     else
        //         return epsilon * (abs(dist) - epsilon / 2);
        // };
        // auto m_error = [&](const Vector2 & point, const Line & line) {
        //     double dist = euclidean_distance(point, line);

        //     if (sq(dist) < sq(epsilon))
        //         return sq(dist);
        //     else
        //         return sq(epsilon);
        // };
        // auto error = m_error;
        // auto best_inlier_indices = decltype(indices)();
        // Vector2 best_eye_center_proj;// = nearest_intersect(pupil_gazelines_proj);
        // double best_line_distance_error = std::numeric_limits<double>::infinity();// = fun::sum(LAMBDA(const Line& line)(error(best_eye_center_proj,line)), pupil_gazelines_proj);

        // for (int i = 0; i < k; ++i) {
        //     auto index_sample = singleeyefitter::randomSubset(indices, n);
        //     auto sample = fun::map([&](size_t i) { return pupil_gazelines_proj[i]; }, index_sample);
        //     auto sample_center_proj = nearest_intersect(sample);
        //     auto index_inliers = fun::filter(
        //     [&](size_t i) { return euclidean_distance(sample_center_proj, pupil_gazelines_proj[i]) < epsilon; },
        //     indices);
        //     auto inliers = fun::map([&](size_t i) { return pupil_gazelines_proj[i]; }, index_inliers);

        //     if (inliers.size() <= w * pupil_gazelines_proj.size()) {
        //         continue;
        //     }

        //     auto inlier_center_proj = nearest_intersect(inliers);
        //     double line_distance_error = fun::sum(
        //     [&](size_t i) { return error(inlier_center_proj, pupil_gazelines_proj[i]); },
        //     indices);

        //     if (line_distance_error < best_line_distance_error) {
        //         best_eye_center_proj = inlier_center_proj;
        //         best_line_distance_error = line_distance_error;
        //         best_inlier_indices = std::move(index_inliers);
        //     }
        // }

        // // std::cout << "Inliers: " << best_inlier_indices.size()
        // //     << " (" << (100.0*best_inlier_indices.size() / pupil_gazelines_proj.size()) << "%)"
        // //     << " = " << best_line_distance_error
        // //     << std::endl;

        // for (auto& pupil : pupils) {
        //     pupil.init_valid = false;
        // }

        // for (auto& i : best_inlier_indices) {
        //     pupils[i].init_valid = true;
        // }

        // if (best_inlier_indices.size() > 0) {
        //     eye_center_proj = best_eye_center_proj;
        //     valid_eye = true;

        // } else {
        //     valid_eye = false;
        // }

    } else {
        // for (auto& pupil : pupils) {
        //     pupil.init_valid = true;
        // }

        eye_center_proj = nearest_intersect(pupil_gazelines_proj);
        valid_eye = true;
    }

    pupils.resize(mObservations.size() );

    if (valid_eye) {
        mSphere.center << eye_center_proj* eye_z / mFocalLength,
                   eye_z;
        mSphere.radius = 1;

        // Disambiguate pupil circles using projected eyeball center
        //
        // Assume that the gaze vector points away from the eye center, and
        // so projected gaze points away from projected eye center. Pick the
        // solution which satisfies this assumption

        //TODO we should be able to move this to initmodel
        for (size_t i = 0; i < mObservations.size(); ++i) {
            const auto& pupil_pair = mObservations[i]->mUnprojectedCirclePair;
            const auto& line = pupil_gazelines_proj[i];
            const auto& c_proj = line.origin();
            const auto& v_proj = line.direction();

            // Check if v_proj going away from est eye center. If it is, then
            // the first circle was correct. Otherwise, take the second one.
            // The two normals will point in opposite directions, so only need
            // to check one.
            if ((c_proj - eye_center_proj).dot(v_proj) >= 0) {
                pupils[i].circle =  pupil_pair.first;

            } else {
                pupils[i].circle = pupil_pair.second;
            }


            // calculate the center variance of the projected gaze vectors to the current eye center
          //  center_distance_variance += euclidean_distance_squared( eye.center, Line3(pupils[i].circle.center, pupils[i].circle.normal ) );

        }
        //center_distance_variance /= pupils.size();
        //std::cout << "center distance variance " << center_distance_variance << std::endl;

    } else {
        // No inliers, so no eye
        mSphere = Sphere::Null;

        // Arbitrarily pick first circle
        // for (size_t i = 0; i < pupils.size(); ++i) {
        //     const auto& pupil_pair = pupil_unprojection_pairs[i];
        //     pupils[i].circle = std::move(pupil_pair.first);
        // }
    }

    return std::move(pupils);

}

std::vector<EyeModel::CircleDescription> EyeModel::initialiseModel( std::vector<EyeModel::CircleDescription>& pupils){

    std::lock_guard<std::mutex> lockObservations(mObservationMutex);

    if (mSphere == Sphere::Null) {
        return pupils;
    }
    std::cout << "init model" << std::endl;

    // Find pupil positions on eyeball to get radius
    //
    // For each image, calculate the 'most likely' position of the pupil
    // circle given the eyeball sphere estimate and gaze vector. Re-estimate
    // the gaze vector to be consistent with this position.
    // First estimate of pupil center, used only to get an estimate of eye radius
    double eye_radius_acc = 0;
    int eye_radius_count = 0;

    for (const auto& pupil : pupils) {

        // Intersect the gaze from the eye center with the pupil circle
        // center projection line (with perfect estimates of gaze, eye
        // center and pupil circle center, these should intersect,
        // otherwise find the nearest point to both lines)
        Vector3 pupil_center = nearest_intersect(Line3(mSphere.center, pupil.circle.normal),
                               Line3(mCameraCenter, pupil.circle.center.normalized()));
        auto distance = (pupil_center - mSphere.center).norm();
        eye_radius_acc += distance;
        ++eye_radius_count;
    }

    // Set the eye radius as the mean distance from pupil centers to eye center
    mSphere.radius = eye_radius_acc / eye_radius_count;

    // Second estimate of pupil radius, used to get position of pupil on eye

    //TODO do we really need this if we don't do refinement ?????
    for (auto& pupil : pupils) {
        pupil = getIntersectedCircle(pupil.circle);
    }

    // Scale eye to anthropomorphic average radius of 12mm
    auto scale = 12.0 / mSphere.radius;
    mSphere.radius = 12.0;
    mSphere.center *= scale;

    double center_distance_variance = 0;
    for (auto& pupil : pupils) {

        pupil.params.radius *= scale;
        pupil.circle = circleFromParams(pupil.params);

         // calculate the center variance of the projected gaze vectors to the current eye center
        //center_distance_variance += euclidean_distance_squared( mSphere.center, Line3(pupil.circle.center, pupil.unprojected_circle.normal ) );
    }
    return pupils;
    //center_distance_variance /= eye_radius_count;
    //std::cout << "center distance variance " << center_distance_variance << std::endl;

    //mLatestPupil = pupils.back();
    // Try previous circle in case of bad fits
    /*EllipseGoodnessFunction<double> goodnessFunction;
    for (int i = 1; i < pupils.size(); ++i) {
    auto& pupil = pupils[i];
    auto& prevPupil = pupils[i-1];

    if (prevPupil.circle) {
    double currentGoodness, prevGoodness;
    if (pupil.circle) {
    currentGoodness = goodnessFunction(eye, pupil.params.theta, pupil.params.psi, pupil.params.radius, focal_length, pupil.observation.image);
    prevGoodness = goodnessFunction(eye, prevPupil.params.theta, prevPupil.params.psi, prevPupil.params.radius, focal_length, pupil.observation.image);
    }

    if (!pupil.circle || prevGoodness > currentGoodness) {
    pupil.circle = prevPupil.circle;
    pupil.params = prevPupil.params;
    }
    }
    }*/





}

void EyeModel::refineWithEdges(std::vector<EyeModel::CircleDescription>& pupils)
{
    int current_model_version;
    Eigen::Matrix<double, Eigen::Dynamic, 1> x;
    x = Eigen::Matrix<double, Eigen::Dynamic, 1>(3 + 3 * pupils.size());
    x.segment<3>(0) = mSphere.center;
    for (int i = 0; i < pupils.size(); ++i) {
        const CircleParams& pupil_params = pupils[i].params;
        x.segment<3>(3 + 3 * i)[0] = pupil_params.theta;
        x.segment<3>(3 + 3 * i)[1] = pupil_params.psi;
        x.segment<3>(3 + 3 * i)[2] = pupil_params.radius;
    }


    ceres::Problem problem;
    std::lock_guard<std::mutex> lockObservations(mObservationMutex);
    {
        for (int i = 0; i < pupils.size(); ++i) {
           /* const cv::Mat& eye_image = pupils[i].observation.image;*/
            const auto& pupil_inliers = mObservations[i]->mObservation2D->final_edges;

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<EllipseDistanceResidualFunction<double>, ceres::DYNAMIC, 3, 3>(
                new EllipseDistanceResidualFunction<double>(/*eye_image,*/ pupil_inliers, mSphere.radius, mFocalLength),
                pupil_inliers.size()
                ),
                NULL, &x[0], &x[3 + 3 * i]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1000;
    options.function_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = true;
    options.update_state_every_iteration = true;
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
    std::cout << summary.BriefReport() << "\n";

    {
        //std::lock_guard<std::mutex> lock_model(model_mutex);
        mSphere.center = x.segment<3>(0);

        // for (int i = 0; i < pupils.size(); ++i) {
        //     auto&& pupil_param = x.segment<3>(3 + 3 * i);
        //     pupils[i].params = PupilParams(pupil_param[0], pupil_param[1], pupil_param[2]);
        //     pupils[i].circle = circleFromParams(eye, pupils[i].params);
        // }
    }
}

void EyeModel::transferNewObservations(){
    std::lock_guard<std::mutex> lock(mObservationMutex); // this call blocks until we get a the lock
    mObservations.insert( mObservations.cend(), mNewObservations.begin(), mNewObservations.end() );
}

double EyeModel::getModelSupport(const Circle&  unprojectedCircle, const Circle& initialisedCircle) const {

    // the angle between the unprojected and the initialised circle normal tells us how good the current observation supports our current model
    // if our model is good and the camera didn't change the perspective or so, these normals should align pretty well
    const auto& n1 = unprojectedCircle.normal;
    const auto& n2 = initialisedCircle.normal;
    const double normals_angle = n1.dot(n2);
    return normals_angle;
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
    const int BIN_AMOUNTS = 20;
    Vector3 pupil_normal =  circle.normal; // the same as a vector from unit sphere center to the pupil center

    const double bin_width = 1.0 / BIN_AMOUNTS;
    // calculate bin
    // values go from -1 to 1
    double x = pupil_normal.x();
    double y = pupil_normal.y();
    x = math::round(x , bin_width);
    y = math::round(y , bin_width);

    Vector2 bin(x, y);
    auto search = mSpatialBins.find(bin);

    if (search == mSpatialBins.end() || search->second == false) {

        // there is no bin at this coord or it is empty
        // so add one
        mSpatialBins.emplace(bin, true);
        double z = std::copysign(std::sqrt(1.0 - x * x - y * y),  pupil_normal.z());
        Vector3 bin_positions_3d(x , y, z); // for visualization
        mBinPositions.push_back(std::move(bin_positions_3d));
        return true;
    }

    return false;

}



const Circle& EyeModel::selectUnprojectedCircle( const std::pair<const Circle, const Circle>& circles) const
{
    if (mSphere == Sphere::Null) {
        throw std::runtime_error("Need to get eye center estimate first (by unprojecting multiple observations)");
    }

    const Vector3& c = circles.first.center;
    const Vector3& v = circles.first.normal;
    Vector2 c_proj = project(c, mFocalLength);
    Vector2 v_proj = project(v + c, mFocalLength) - c_proj;
    v_proj.normalize();
    Vector2 eye_center_proj = project(mSphere.center, mFocalLength);

    if ((c_proj - eye_center_proj).dot(v_proj) >= 0) {
        return circles.first;

    } else {
       return circles.second;
    }

}

EyeModel::CircleDescription EyeModel::getIntersectedCircle( const Circle& unprojectedCircle) const
{
    // Ignore the circle normal, and intersect the circle
    // center projection line with the sphere
    CircleDescription circleDesc;
    try {
        auto pupil_center_sphere_intersect = intersect(Line3(mCameraCenter, unprojectedCircle.center.normalized()),
                                             mSphere);
        auto new_pupil_center = pupil_center_sphere_intersect.first;
        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = unprojectedCircle.radius / unprojectedCircle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_center.z();
        // Parametrise this new pupil position using spherical coordinates
        Vector3 center_to_pupil = new_pupil_center - mSphere.center;
        double r = center_to_pupil.norm();
        circleDesc.params.theta = acos(center_to_pupil[1] / r);
        circleDesc.params.psi = atan2(center_to_pupil[2], center_to_pupil[0]);
        circleDesc.params.radius = new_pupil_radius;
        // Update pupil circle to match parameters
        circleDesc.circle = circleFromParams( circleDesc.params );

    } catch (no_intersection_exception&) {
        circleDesc.circle =  Circle::Null;
        circleDesc.params = CircleParams();
    }

    return circleDesc;
}

Circle EyeModel::circleFromParams(const Sphere& eye, CircleParams& params) const
{
    if (params.radius == 0)
        return Circle::Null;

    Vector3 radial = math::sph2cart<double>(double(1), params.theta, params.psi);
    return Circle(eye.center + eye.radius * radial,
                  radial,
                  params.radius);
}

Circle EyeModel::circleFromParams(CircleParams& params) const
{
    return circleFromParams(mSphere, params);
}

} // singleeyefitter
