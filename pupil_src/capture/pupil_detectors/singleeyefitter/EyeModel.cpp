

#include "EyeModel.h"


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




Circle EyeModel::presentObservation(const ObservationPtr observation)
{

    Circle& unprojected_circle;
    Circle initialised_circle;
    bool should_add_observation = false;

    if (mSphere != Sphere::Null) {
        // select the right circle depending on the current model
        unprojected_circle = selectUnprojectedCircle(observation->mUnprojectedCirclePair);
        // initialised circle. circle parameters addapted to our current eye model
        initialised_circle = getIntersectedCircle(unprojected_circle);

    }

    //Check for properties if it's a candidate we can use
    if (mSphere != Sphere::Null) {

        if (unprojected_circle != Circle::Null && initialised_circle != Circle::Null) {  // initialise failed

            double support = getModelSupport(unprojected_circle, initialised_circle);

            //std::cout << "support: " << support  << std::endl;
            if (support > 0.97) {

                if (isSpatialRelevant(initialised_circle)) {
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
        mObservations.push_back(std::move(observation));

        std::cout << "pupil size " << pupils.size() << std::endl;

        //refine model every 50 new pupils
        if (mObservations.size() > 50 && mObservations.size() % 50  == 0) {

            unproject_observations();
            initialiseModel();

            std::cout << "-----------refine model"  << std::endl;
            std::cout << "-----------prev eye: " << eye << std::endl;
            refineWithEdges();
            std::cout << "-----------new eye: " << eye << std::endl;

        } else if (pupils.size() <= 50) {

            unproject_observations();
            initialiseModel();
        }

    } else {

        // if we don't add a new one we still wanna have the latest pupil parameters
        mLatestPupil = std::move(initialised_circle);
    }

}

void EyeModelFitter::unprojectObservations( bool use_ransac /*= true*/)
{
    using math::sq;
    std::lock_guard<std::mutex> lock_model(model_mutex);

    if (pupils.size() < 2) {
        return;
    }
    const double pupil_radius = 1;
    const eye_z = 57;
    //std::vector<std::pair<Circle, Circle>> pupil_unprojection_pairs;
    std::vector<Line> pupil_gazelines_proj;

    for (const auto& pupil : pupils) {
        // Get pupil circles (up to depth)
        //
        // Do a per-image unprojection of the pupil ellipse into the two fixed
        // size circles that would project onto it. The size of the circles
        // doesn't matter here, only their center and normal does.
        auto unprojection_pair = unproject(pupil.observation->ellipse,
                                           pupil_radius, focal_length);
        // Get projected circles and gaze vectors
        //
        // Project the circle centers and gaze vectors down back onto the image
        // plane. We're only using them as line parametrisations, so it doesn't
        // matter which of the two centers/gaze vectors we use, as the
        // two gazes are parallel and the centers are co-linear.
        const auto& c = unprojection_pair.first.center;
        const auto& v = unprojection_pair.first.normal;
        Vector2 c_proj = project(c, focal_length);
        Vector2 v_proj = project(v + c, focal_length) - c_proj;
        v_proj.normalize();
        pupil_unprojection_pairs.push_back(std::move(unprojection_pair));
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

    if (use_ransac) {
        auto indices = fun::range_<std::vector<size_t>>(pupil_gazelines_proj.size());
        const int n = 2;
        double w = 0.3;
        double p = 0.9999;
        int k = ceil(log(1 - p) / log(1 - pow(w, n)));
        double epsilon = 10;
        auto huber_error = [&](const Vector2 & point, const Line & line) {
            double dist = euclidean_distance(point, line);

            if (sq(dist) < sq(epsilon))
                return sq(dist) / 2;
            else
                return epsilon * (abs(dist) - epsilon / 2);
        };
        auto m_error = [&](const Vector2 & point, const Line & line) {
            double dist = euclidean_distance(point, line);

            if (sq(dist) < sq(epsilon))
                return sq(dist);
            else
                return sq(epsilon);
        };
        auto error = m_error;
        auto best_inlier_indices = decltype(indices)();
        Vector2 best_eye_center_proj;// = nearest_intersect(pupil_gazelines_proj);
        double best_line_distance_error = std::numeric_limits<double>::infinity();// = fun::sum(LAMBDA(const Line& line)(error(best_eye_center_proj,line)), pupil_gazelines_proj);

        for (int i = 0; i < k; ++i) {
            auto index_sample = singleeyefitter::randomSubset(indices, n);
            auto sample = fun::map([&](size_t i) { return pupil_gazelines_proj[i]; }, index_sample);
            auto sample_center_proj = nearest_intersect(sample);
            auto index_inliers = fun::filter(
            [&](size_t i) { return euclidean_distance(sample_center_proj, pupil_gazelines_proj[i]) < epsilon; },
            indices);
            auto inliers = fun::map([&](size_t i) { return pupil_gazelines_proj[i]; }, index_inliers);

            if (inliers.size() <= w * pupil_gazelines_proj.size()) {
                continue;
            }

            auto inlier_center_proj = nearest_intersect(inliers);
            double line_distance_error = fun::sum(
            [&](size_t i) { return error(inlier_center_proj, pupil_gazelines_proj[i]); },
            indices);

            if (line_distance_error < best_line_distance_error) {
                best_eye_center_proj = inlier_center_proj;
                best_line_distance_error = line_distance_error;
                best_inlier_indices = std::move(index_inliers);
            }
        }

        // std::cout << "Inliers: " << best_inlier_indices.size()
        //     << " (" << (100.0*best_inlier_indices.size() / pupil_gazelines_proj.size()) << "%)"
        //     << " = " << best_line_distance_error
        //     << std::endl;

        for (auto& pupil : pupils) {
            pupil.init_valid = false;
        }

        for (auto& i : best_inlier_indices) {
            pupils[i].init_valid = true;
        }

        if (best_inlier_indices.size() > 0) {
            eye_center_proj = best_eye_center_proj;
            valid_eye = true;

        } else {
            valid_eye = false;
        }

    } else {
        for (auto& pupil : pupils) {
            pupil.init_valid = true;
        }

        eye_center_proj = nearest_intersect(pupil_gazelines_proj);
        valid_eye = true;
    }

    if (valid_eye) {
        eye.center << eye_center_proj* eye_z / focal_length,
                   eye_z;
        eye.radius = 1;

        // Disambiguate pupil circles using projected eyeball center
        //
        // Assume that the gaze vector points away from the eye center, and
        // so projected gaze points away from projected eye center. Pick the
        // solution which satisfies this assumption
        for (size_t i = 0; i < pupils.size(); ++i) {
            const auto& pupil_pair = pupil_unprojection_pairs[i];
            const auto& line = pupil_gazelines_proj[i];
            const auto& c_proj = line.origin();
            const auto& v_proj = line.direction();

            // Check if v_proj going away from est eye center. If it is, then
            // the first circle was correct. Otherwise, take the second one.
            // The two normals will point in opposite directions, so only need
            // to check one.
            if ((c_proj - eye_center_proj).dot(v_proj) >= 0) {
                pupils[i].circle = std::move(pupil_pair.first);

            } else {
                pupils[i].circle = std::move(pupil_pair.second);
            }

             pupils[i].unprojected_circle = pupils[i].circle; // keep track of the unprojected one

            // calculate the center variance of the projected gaze vectors to the current eye center
          //  center_distance_variance += euclidean_distance_squared( eye.center, Line3(pupils[i].circle.center, pupils[i].circle.normal ) );

        }
        //center_distance_variance /= pupils.size();
        //std::cout << "center distance variance " << center_distance_variance << std::endl;

    } else {
        // No inliers, so no eye
        eye = Sphere::Null;

        // Arbitrarily pick first circle
        for (size_t i = 0; i < pupils.size(); ++i) {
            const auto& pupil_pair = pupil_unprojection_pairs[i];
            pupils[i].circle = std::move(pupil_pair.first);
        }
    }

}

double EyeModelFitter::getModelSupport(const Circle&  unprojectedCircle, const Circle& initialisedCircle) const {

    // the angle between the unprojected and the initialised circle normal tells us how good the current observation supports our current model
    // if our model is good and the camera didn't change the perspective or so, these normals should align pretty well
    const auto& n1 = unprojectedCircle.normal;
    const auto& n2 = initialisedCircle.normal;
    const double normals_angle = n1.dot(n2);
    return normals_angle;
}

bool EyeModelFitter::isSpatialRelevant(const Circle& circle) const {

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
    auto search = pupil_position_bins.find(bin);

    if (search == pupil_position_bins.end() || search->second == false) {

        // there is no bin at this coord or it is empty
        // so add one
        pupil_position_bins.emplace(bin, true);
        double z = std::copysign(std::sqrt(1.0 - x * x - y * y),  pupil_normal.z());
        Vector3 bin_positions_3d(x , y, z); // for visualization
        bin_positions.push_back(std::move(bin_positions_3d));
        return true;
    }

    return false;

}



const Circle& EyeModelFitter::selectUnprojectedCircle( const std::pair<const Circle, const Circle>& circles) const
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

Circle EyeModelFitter::getIntersectedCircle( const Circle& unprojectedCircle)
{
    // Ignore the circle normal, and intersect the circle
    // center projection line with the sphere
    try {
        auto pupil_center_sphere_intersect = intersect(Line3(mCameraCenter, pupil.circle.center.normalized()),
                                             mSpehre);
        auto new_pupil_center = pupil_center_sphere_intersect.first;
        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = pupil.circle.radius / pupil.circle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_center.z();
        // Parametrise this new pupil position using spherical coordinates
        Vector3 center_to_pupil = new_pupil_center - mSpehre.center;
        double r = center_to_pupil.norm();
        pupil.params.theta = acos(center_to_pupil[1] / r);
        pupil.params.psi = atan2(center_to_pupil[2], center_to_pupil[0]);
        pupil.params.radius = new_pupil_radius;
        // Update pupil circle to match parameters
        pupil.circle = circleFromParams(pupil.params);

    } catch (no_intersection_exception&) {
        pupil.circle = Circle::Null;
        pupil.params.theta = 0;
        pupil.params.psi = 0;
        pupil.params.radius = 0;
    }

    return pupil.circle;
}

} // singleeyefitter
