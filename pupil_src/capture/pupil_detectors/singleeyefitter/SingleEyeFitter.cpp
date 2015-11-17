// SingleEyeFitter.cpp : Defines the entry point for the console application.
//


#include <boost/math/special_functions/sign.hpp>
#include <Eigen/StdVector>

#include <queue>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <ceres/jet.h>

#include "singleeyefitter.h"
#include "EllipseDistanceApproxCalculator.h"
#include "EllipseGoodnessFunction.h"
#include "EllipseDistanceResidualFunction.h"
#include "EllipsePointDistanceFunction.h"

#include "Fit/CircleOnSphereFit.h"
#include "CircleDeviationVariance3D.h"
#include "CircleEvaluation3D.h"
#include "CircleGoodness3D.h"

#include "PupilContrastTerm.h"
#include "PupilAnthroTerm.h"

#include "utils.h"
#include "ImageProcessing/cvx.h"
#include "Geometry/Conic.h"
#include "Geometry/Ellipse.h"
#include "Geometry/Circle.h"
#include "Geometry/Conicoid.h"
#include "Geometry/Sphere.h"
#include "solve.h"
#include "intersect.h"
#include "projection.h"
#include "fun.h"

#include "mathHelper.h"
#include "distance.h"
#include "common/traits.h"

#include <spii/spii.h>
#include <spii/term.h>
#include <spii/function.h>
#include <spii/solver.h>



namespace singleeyefitter {

    template<typename Scalar>
    cv::Rect bounding_box(const Ellipse2D<Scalar>& ellipse)
    {
        using std::sin;
        using std::cos;
        using std::sqrt;
        using std::floor;
        using std::ceil;
        Scalar ux = ellipse.major_radius * cos(ellipse.angle);
        Scalar uy = ellipse.major_radius * sin(ellipse.angle);
        Scalar vx = ellipse.minor_radius * cos(ellipse.angle + PI / 2);
        Scalar vy = ellipse.minor_radius * sin(ellipse.angle + PI / 2);
        Scalar bbox_halfwidth = sqrt(ux * ux + vx * vx);
        Scalar bbox_halfheight = sqrt(uy * uy + vy * vy);
        return cv::Rect(floor(ellipse.center[0] - bbox_halfwidth), floor(ellipse.center[1] - bbox_halfheight),
                        2 * ceil(bbox_halfwidth) + 1, 2 * ceil(bbox_halfheight) + 1);
    }

    template<template<class, int> class Jet, class T, int N>
    typename std::enable_if<std::is_same<typename ad_traits<Jet<T, N>>::ad_tag, ceres_jet_tag>::value, Ellipse2D<T>>::type
            toConst(const Ellipse2D<Jet<T, N>>& ellipse)
    {
        return Ellipse2D<T>(
                   ellipse.center[0].a,
                   ellipse.center[1].a,
                   ellipse.major_radius.a,
                   ellipse.minor_radius.a,
                   ellipse.angle.a);
    }

    template<class T>
    Ellipse2D<T> scaledMajorRadius(const Ellipse2D<T>& ellipse, const T& target_radius)
    {
        return Ellipse2D<T>(
                   ellipse.center[0],
                   ellipse.center[1],
                   target_radius,
                   target_radius * ellipse.minor_radius / ellipse.major_radius,
                   ellipse.angle);
    };


    template<typename T>
    T angleDiffGoodness(T theta1, T psi1, T theta2, T psi2, typename ad_traits<T>::scalar sigma)
    {
        using std::sin;
        using std::cos;
        using std::acos;
        using std::asin;
        using std::atan2;
        using std::sqrt;

        if (theta1 == theta2 && psi1 == psi2) {
            return T(1);
        }

        // Haversine distance
        auto dist = T(2) * asin(sqrt(sq(sin((theta1 - theta2) / T(2))) + cos(theta1) * cos(theta2) * sq(sin((psi1 - psi2) / T(2)))));
        return exp(-sq(dist) / sq(sigma));
    }

    template<typename T>
    Circle3D<T> circleOnSphere(const Sphere<T>& sphere, T theta, T psi, T circle_radius)
    {
        typedef Eigen::Matrix<T, 3, 1> Vector3;
        Vector3 radial = math::sph2cart<T>(T(1), theta, psi);
        return Circle3D<T>(sphere.center + sphere.radius * radial,
                           radial,
                           circle_radius);
    }





    const Vector3 EyeModelFitter::camera_center = Vector3::Zero();


    EyeModelFitter::Pupil::Pupil(std::shared_ptr<Detector_2D_Results> observation) : observation(observation), params(0, 0, 0)
    {
    }

    EyeModelFitter::Pupil::Pupil()
    {
    }


    EyeModelFitter::PupilParams::PupilParams(double theta, double psi, double radius) : theta(theta), psi(psi), radius(radius)
    {
    }

    EyeModelFitter::PupilParams::PupilParams() : theta(0), psi(0), radius(0)
    {
    }


    // EyeModelFitter::Observation::Observation(/*cv::Mat image, */Ellipse ellipse/*, std::vector<cv::Point2f> inliers*/,   std::vector<std::vector<cv::Point2i>> contours) : /* image(image),*/ ellipse(ellipse)/*, inliers(std::move(inliers)*/, contours(contours)
    // {
    //     // for(auto& contour : contours){
    //     //     for( int i =0 ; i < contour.size(); i++){
    //     //         std::cout << "[" << contour[i].x << " " << contour[i].y << "] ";
    //     //     }
    //     //     std::cout << std::endl;
    //     // }
    // }

    // EyeModelFitter::Observation::Observation()
    // {
    // }

}


singleeyefitter::EyeModelFitter::EyeModelFitter() : region_band_width(5), region_step_epsilon(0.5), region_scale(1), max_pupils(300)
{
}
singleeyefitter::EyeModelFitter::EyeModelFitter(double focal_length, double region_band_width, double region_step_epsilon) : focal_length(focal_length), region_band_width(region_band_width), region_step_epsilon(region_step_epsilon), region_scale(1), max_pupils(300)
{
}

// singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_observation(cv::Mat image, Ellipse pupil, int n_pseudo_inliers /*= 0*/)
// {
//     std::vector<cv::Point2f> pupil_inliers;
//     for (int i = 0; i < n_pseudo_inliers; ++i) {
//         auto p = pointAlongEllipse(pupil, i * 2 * M_PI / n_pseudo_inliers);
//         pupil_inliers.emplace_back(static_cast<float>(p[0]), static_cast<float>(p[1]));
//     }
//     return add_observation(std::move(image), std::move(pupil), std::move(pupil_inliers));
// }

// singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_observation(cv::Mat image, Ellipse pupil, std::vector<cv::Point2f> pupil_inliers)
// {
//     assert(image.channels() == 1 && image.depth() == CV_8U);

//     std::lock_guard<std::mutex> lock_model(model_mutex);

//     pupils.emplace_back(
//         Observation(std::move(image), std::move(pupil), std::move(pupil_inliers))
//         );
//     return pupils.size() - 1;
// }

// singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_observation( Ellipse pupil )
// {
//     // std::vector<cv::Point2f> pupil_inliers;
//     // for (int i = 0; i < n_pseudo_inliers; ++i) {
//     //     auto p = pointAlongEllipse(pupil, i * 2 * M_PI / n_pseudo_inliers);
//     //     pupil_inliers.emplace_back(static_cast<float>(p[0]), static_cast<float>(p[1]));
//     // }
//     std::lock_guard<std::mutex> lock_model(model_mutex);

//     pupils.emplace_back(
//         Observation(/*std::move(image), */std::move(pupil)/*, std::move(pupil_inliers)*/, {} )
//         );
//     return pupils.size() - 1;
// }

// singleeyefitter::EyeModelFitter::Index singleeyefitter::EyeModelFitter::add_observation(Ellipse pupil, std::vector<int32_t*> contour_ptrs , std::vector<size_t> contour_sizes)
// {
//     std::lock_guard<std::mutex> lock_model(model_mutex);
//     // uint i = 0;
//     // for( auto* contour : contour_ptrs){
//     //     uint length = contour_sizes.at(i);
//     //     for( int k = 0; k<length; k+=2){
//     //         std::cout << "[" << contour[k] << ","<< contour[k+1] << "] " ;
//     //     }
//     //     std::cout << std::endl;
//     //     i++;
//     // }
//     std::vector<std::vector<cv::Point2i>> contours;
//     int i = 0;

//     for (int32_t* contour_ptr : contour_ptrs) {
//         uint length = contour_sizes.at(i);
//         std::vector<cv::Point2i> contour;

//         for (int j = 0; j < length; j += 2) {
//             contour.emplace_back(contour_ptr[j], contour_ptr[j + 1]);
//         }

//         contours.push_back(contour);
//         i++;
//     }

//     pupils.emplace_back(
//         Observation(/*std::move(image), */std::move(pupil)/*, std::move(pupil_inliers)*/, std::move(contours))
//     );
//     return pupils.size() - 1;
// }

void singleeyefitter::EyeModelFitter::update(std::shared_ptr<Detector_2D_Results>& observation)
{


    //check first if the observations is even strong enough to be added
    if (observation->confidence  <  0.9)
        return; // too weak


    //if the observation is strong enough, check for other properties if it's a candidate we can you




    //if the observation passed all tests we can add it
    add_observation(observation);
    // when ever we add a new observation we need to rebuild the eye model
    unproject_observations();
    initialise_model();



}


singleeyefitter::Index singleeyefitter::EyeModelFitter::add_observation(std::shared_ptr<Detector_2D_Results>& observation)
{
    std::lock_guard<std::mutex> lock_model(model_mutex);

    while (pupils.size() >= max_pupils) pupils.pop_front();

    // uint i = 0;
    // for( auto* contour : contour_ptrs){
    //     uint length = contour_sizes.at(i);
    //     for( int k = 0; k<length; k+=2){
    //         std::cout << "[" << contour[k] << ","<< contour[k+1] << "] " ;
    //     }
    //     std::cout << std::endl;
    //     i++;
    // }

    // observations are realtive to their ROI !!!
    cv::Rect roi = observation->current_roi;
    int image_height = observation->image_height;
    int image_width = observation->image_width;

    for (Contour_2D& c : observation->final_contours) {
        for (cv::Point& p : c) {
            p += roi.tl();
            p.x -= image_width * 0.5f;
            p.y = image_height * 0.5f - p.y;
        }
    }

    for (Contour_2D& c : observation->contours) {
        for (cv::Point& p : c) {
            p += roi.tl();
            p.x -= image_width * 0.5f;
            p.y = image_height * 0.5f - p.y;
        }
    }

    for (cv::Point& p : observation->raw_edges) {
        p += roi.tl();
        p.x -= image_width * 0.5f;
        p.y = image_height * 0.5f - p.y;
    }

    Ellipse& ellipse = observation->ellipse;
    ellipse.center[0] += roi.x;
    ellipse.center[1] += roi.y;
    ellipse.center[0] -= image_width * 0.5f;
    ellipse.center[1] = image_height * 0.5f - ellipse.center[1];
    ellipse.angle = -ellipse.angle; //take y axis flip into account


    auto pupil = Pupil(observation);

    // we decide here if the observation will be added

    if (eye != Sphere::Null) {


        Circle unprojected_circle = unproject_single_observation(pupil, 5); // unproject circle in 3D space, doesn't consider current eye model (cone unprojection of ellipse)
        Circle initialised_circle = initialise_single_observation(pupil);  // initialised circle. circle parameters addapted to our current eye model

        if (unprojected_circle != Circle::Null && initialised_circle != Circle::Null ) { // initialise failed

            // the angle between the unprojected and the initialised circle normal tell us how good the current observation supports our current model
            // if our model is good and the camera didn't change the perspective or so, these normals should align pretty well
            double normals_angle = std::acos(unprojected_circle.normal.dot(initialised_circle.normal));

            if(normals_angle >  0.2 ){ // detected camera slip
                return pupils.size() - 1;
                std::cout << "slip detected"  << std::endl;
            }


            double psi_variance = std::pow(pupil.params.psi - psi_mean , 2);
            double theta_variance = std::pow(pupil.params.theta - theta_mean , 2);

            //std::cout << "psi variance: " <<  psi_variance<< std::endl;
            //std::cout << "theta variance: " <<  theta_variance<< std::endl;

            int bin_amount = 20;
            Vector3 pupil_normal =  pupil.circle.normal; // the same as a vector from unit sphere center to the pupil center

            double bin_width = 1.0 / bin_amount;
            // calculate bin
            // values go from -1 to 1
            double x = pupil_normal.x();//+1.0;// map them to [0,2]
            double y = pupil_normal.y();//+1.0;
            x = math::round(x , bin_width);
            y = math::round(y , bin_width);

            Vector2 bin(x, y);
            auto search = pupil_position_bins.find(bin);

            if (search == pupil_position_bins.end() || search->second == false) {

                // there is no bin at this coord or it is empty
                // so add one
                pupil_position_bins.emplace(bin, true);
                double z = std::copysign(std::sqrt(1.0 - x * x - y * y),  pupil_normal.z());

                Vector3 bin_positions_3d(x , y, z);
                //bin_positions_3d.normalize();
                bin_positions.push_back(bin_positions_3d);

                pupils.push_back(pupil);

            }
        }else{
            std::cout << "no valid circles"  << std::endl;
        }


    } else {
        std::cout << "add without check" << std::endl;
        pupils.push_back(pupil);

    }

    return pupils.size() - 1;
}

void EyeModelFitter::reset()
{
    std::lock_guard<std::mutex> lock_model(model_mutex);
    pupils.clear();
    pupil_position_bins.clear();
    eye = Sphere::Null;
    model_version = 0;
}

singleeyefitter::Circle singleeyefitter::EyeModelFitter::circleFromParams(const Sphere& eye, const PupilParams& params)
{
    if (params.radius == 0)
        return Circle::Null;

    Vector3 radial = math::sph2cart<double>(double(1), params.theta, params.psi);
    return Circle(eye.center + eye.radius * radial,
                  radial,
                  params.radius);
}

singleeyefitter::Circle singleeyefitter::EyeModelFitter::circleFromParams(const PupilParams& params) const
{
    return circleFromParams(eye, params);
}

// void singleeyefitter::EyeModelFitter::print_single_contrast_metric(const Pupil& pupil) const
// {
//     if (!pupil.circle) {
//         std::cout << "No pupil" << std::endl;
//         return;
//     }

//     double params[3];
//     params[0] = pupil.params.theta;
//     params[1] = pupil.params.psi;
//     params[2] = pupil.params.radius;
//     double* vars[1];
//     vars[0] = params;

//     std::vector<Eigen::VectorXd> gradient;
//     gradient.push_back(Eigen::VectorXd::Zero(3));

//     PupilContrastTerm<false> contrast_term(
//         eye,
//         focal_length * region_scale,
//         cvx::resize(pupil.observation.image, region_scale),
//         region_band_width,
//         region_step_epsilon);

//     double contrast_val = contrast_term.evaluate(vars, &gradient);

//     std::cout << "Contrast term: " << contrast_val << std::endl;
//     std::cout << "     gradient: [ " << gradient[0].transpose() << " ]" << std::endl;
// }

// void singleeyefitter::EyeModelFitter::print_single_contrast_metric(Index id) const
// {
//     print_single_contrast_metric(pupils[id]);
// }

// double singleeyefitter::EyeModelFitter::single_contrast_metric(const Pupil& pupil) const
// {
//     if (!pupil.circle) {
//         std::cout << "No pupil" << std::endl;
//         return 0;
//     }

//     double params[3];
//     params[0] = pupil.params.theta;
//     params[1] = pupil.params.psi;
//     params[2] = pupil.params.radius;
//     double* vars[1];
//     vars[0] = params;

//     PupilContrastTerm<false> contrast_term(
//         eye,
//         focal_length * region_scale,
//         cvx::resize(pupil.observation.image, region_scale),
//         region_band_width,
//         region_step_epsilon);

//     double contrast_val = contrast_term.evaluate(vars);

//     return contrast_val;
// }

// double singleeyefitter::EyeModelFitter::single_contrast_metric(Index id) const
// {
//     return single_contrast_metric(pupils[id]);
// }

// const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::refine_single_with_contrast(Pupil& pupil)
// {
//     if (!pupil.circle)
//         return pupil.circle;

//     double params[3];
//     params[0] = pupil.params.theta;
//     params[1] = pupil.params.psi;
//     params[2] = pupil.params.radius;

//     spii::Function f;
//     f.add_variable(&params[0], 3);
//     f.add_term(std::make_shared<PupilContrastTerm<false>>(
//         eye,
//         focal_length * region_scale,
//         cvx::resize(pupil.observation.image, region_scale),
//         region_band_width,
//         region_step_epsilon), &params[0]);

//     spii::LBFGSSolver solver;
//     solver.log_function = [](const std::string&) {};
//     //solver.function_improvement_tolerance = 1e-5;
//     spii::SolverResults results;
//     solver.solve(f, &results);
//     //std::cout << results << std::endl;

//     pupil.params = PupilParams(params[0], params[1], params[2]);
//     pupil.circle = circleFromParams(pupil.params);

//     return pupil.circle;
// }

// const singleeyefitter::EyeModelFitter::Circle& singleeyefitter::EyeModelFitter::refine_single_with_contrast(Index id)
// {
//     return refine_single_with_contrast(pupils[id]);
// }

const singleeyefitter::Circle& singleeyefitter::EyeModelFitter::initialise_single_observation(Pupil& pupil)
{
    // Ignore the pupil circle normal, and intersect the pupil circle
    // center projection line with the eyeball sphere
    try {
        auto pupil_center_sphere_intersect = intersect(Line3(camera_center, pupil.circle.center.normalized()),
                                             eye);
        auto new_pupil_center = pupil_center_sphere_intersect.first;
        // Now that we have 3D positions for the pupil (rather than just a
        // projection line), recalculate the pupil radius at that position.
        auto pupil_radius_at_1 = pupil.circle.radius / pupil.circle.center.z();
        auto new_pupil_radius = pupil_radius_at_1 * new_pupil_center.z();
        // Parametrise this new pupil position using spherical coordinates
        Vector3 center_to_pupil = new_pupil_center - eye.center;
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

const singleeyefitter::Circle& singleeyefitter::EyeModelFitter::initialise_single_observation(Index id)
{
    initialise_single_observation(pupils[id]);
    /*if (id > 0 && pupils[id-1].circle) {
    // Try previous circle in case of bad fits
    EllipseGoodnessFunction<double> goodnessFunction;
    auto& pupil = pupils[id];
    auto& prevPupil = pupils[id-1];

    double currentGoodness, prevGoodness;
    if (pupil.circle) {
    currentGoodness = goodnessFunction(eye, pupil.params.theta, pupil.params.psi, pupil.params.radius, focal_length, pupil.observation.image);
    prevGoodness = goodnessFunction(eye, prevPupil.params.theta, prevPupil.params.psi, prevPupil.params.radius, focal_length, pupil.observation.image);
    }

    if (!pupil.circle || prevGoodness > currentGoodness) {
    pupil.circle = prevPupil.circle;
    pupil.params = prevPupil.params;
    }
    }*/
    return pupils[id].circle;
}

const singleeyefitter::Circle& singleeyefitter::EyeModelFitter::unproject_single_observation(Pupil& pupil, double pupil_radius /*= 1*/) const
{
    if (eye == Sphere::Null) {
        throw std::runtime_error("Need to get eye center estimate first (by unprojecting multiple observations)");
    }

    // Single pupil version of "unproject_observations"
    auto unprojection_pair = unproject(pupil.observation->ellipse, pupil_radius, focal_length);
    const Vector3& c = unprojection_pair.first.center;
    const Vector3& v = unprojection_pair.first.normal;
    Vector2 c_proj = project(c, focal_length);
    Vector2 v_proj = project(v + c, focal_length) - c_proj;
    v_proj.normalize();
    Vector2 eye_center_proj = project(eye.center, focal_length);

    if ((c_proj - eye_center_proj).dot(v_proj) >= 0) {
        pupil.circle = std::move(unprojection_pair.first);

    } else {
        pupil.circle = std::move(unprojection_pair.second);
    }

    return pupil.circle;
}

const singleeyefitter::Circle& singleeyefitter::EyeModelFitter::unproject_single_observation(Index id, double pupil_radius /*= 1*/)
{
    return unproject_single_observation(pupils[id], pupil_radius);
}

// void singleeyefitter::EyeModelFitter::refine_with_inliers(const CallbackFunction& callback /*= CallbackFunction()*/)
// {
//     int current_model_version;
//     Eigen::Matrix<double, Eigen::Dynamic, 1> x;
//     {
//         std::lock_guard<std::mutex> lock_model(model_mutex);

//         current_model_version = model_version;

//         x = Eigen::Matrix<double, Eigen::Dynamic, 1>(3 + 3 * pupils.size());
//         x.segment<3>(0) = eye.center;
//         for (int i = 0; i < pupils.size(); ++i) {
//             const PupilParams& pupil_params = pupils[i].params;
//             x.segment<3>(3 + 3 * i)[0] = pupil_params.theta;
//             x.segment<3>(3 + 3 * i)[1] = pupil_params.psi;
//             x.segment<3>(3 + 3 * i)[2] = pupil_params.radius;
//         }
//     }

//     ceres::Problem problem;
//     for (int i = 0; i < pupils.size(); ++i) {
//         const cv::Mat& eye_image = pupils[i].observation.image;
//         const auto& pupil_inliers = pupils[i].observation.inliers;

//         problem.AddResidualBlock(
//             new ceres::AutoDiffCostFunction<EllipseDistanceResidualFunction<double>, ceres::DYNAMIC, 3, 3>(
//             new EllipseDistanceResidualFunction<double>(eye_image, pupil_inliers, eye.radius, focal_length),
//             pupil_inliers.size()
//             ),
//             NULL, &x[0], &x[3 + 3 * i]);
//     }

//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_SCHUR;
//     options.max_num_iterations = 1000;
//     options.function_tolerance = 1e-10;
//     options.minimizer_progress_to_stdout = true;
//     options.update_state_every_iteration = true;
//     if (callback) {
//         struct CallCallbackWrapper : public ceres::IterationCallback
//         {
//             double eye_radius;
//             const CallbackFunction& callback;
//             const Eigen::Matrix<double, Eigen::Dynamic, 1>& x;

//             CallCallbackWrapper(const EyeModelFitter& fitter, const CallbackFunction& callback, const Eigen::Matrix<double, Eigen::Dynamic, 1>& x)
//                 : eye_radius(fitter.eye.radius), callback(callback), x(x) {}

//             virtual ceres::CallbackReturnType operator() (const ceres::IterationSummary& summary) {
//                 Eigen::Matrix<double, 3, 1> eye_pos(x[0], x[1], x[2]);
//                 Sphere eye(eye_pos, eye_radius);

//                 std::vector<Circle> pupils;
//                 for (int i = 0; i < (x.size() - 3)/3; ++i) {
//                     auto&& pupil_param_v = x.segment<3>(3 + 3 * i);
//                     pupils.push_back(EyeModelFitter::circleFromParams(eye, PupilParams(pupil_param_v[0], pupil_param_v[1], pupil_param_v[2])));
//                 }

//                 callback(eye, pupils);

//                 return ceres::SOLVER_CONTINUE;
//             }
//         };
//         options.callbacks.push_back(new CallCallbackWrapper(*this, callback, x));
//     }
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
//     std::cout << summary.BriefReport() << "\n";

//     {
//         std::lock_guard<std::mutex> lock_model(model_mutex);

//         if (current_model_version != model_version)    {
//             std::cout << "Old model, not applying refined parameters" << std::endl;
//             return;
//         }

//         eye.center = x.segment<3>(0);

//         for (int i = 0; i < pupils.size(); ++i) {
//             auto&& pupil_param = x.segment<3>(3 + 3 * i);
//             pupils[i].params = PupilParams(pupil_param[0], pupil_param[1], pupil_param[2]);
//             pupils[i].circle = circleFromParams(eye, pupils[i].params);
//         }
//     }
// }

// void singleeyefitter::EyeModelFitter::refine_with_region_contrast(const CallbackFunction& callback /*= CallbackFunction()*/)
// {
//     int current_model_version;
//     Eigen::Matrix<double, Eigen::Dynamic, 1> x0;
//     spii::Function f;
//     {
//         std::lock_guard<std::mutex> lock_model(model_mutex);

//         current_model_version = model_version;

//         x0 = Eigen::Matrix<double, Eigen::Dynamic, 1>(3 + 3 * pupils.size());
//         x0.segment<3>(0) = eye.center;
//         for (int i = 0; i < pupils.size(); ++i) {
//             const PupilParams& pupil_params = pupils[i].params;
//             x0.segment<3>(3 + 3 * i)[0] = pupil_params.theta;
//             x0.segment<3>(3 + 3 * i)[1] = pupil_params.psi;
//             x0.segment<3>(3 + 3 * i)[2] = pupil_params.radius;
//         }

//         f.add_variable(&x0[0], 3);
//         for (int i = 0; i < pupils.size(); ++i) {
//             if (pupils[i].circle) {
//                 f.add_variable(&x0[3 + 3 * i], 3);

//                 f.add_term(
//                     std::make_shared<PupilContrastTerm<true>>(
//                     eye,
//                     focal_length * region_scale,
//                     cvx::resize(pupils[i].observation.image, region_scale),
//                     region_band_width,
//                     region_step_epsilon),
//                     &x0[0], &x0[3 + 3 * i]);
//                 //f.add_term(std::make_shared<PupilAnthroTerm>(2.5, 1, 0.001), &x0[3+3*i]);

//                 /*if (i == 0 || !pupils[i-1].circle) {
//                 } else {
//                 vars.push_back(&x0[3+3*(i-1)]);
//                 f.add_term(std::make_shared<SinglePupilTerm<true,true,true>>(*this, pupils[i].observation.image), vars);
//                 }*/
//             }
//         }
//     }

//     spii::LBFGSSolver solver;
//     solver.maximum_iterations = 1000;
//     solver.function_improvement_tolerance = 1e-5;
//     spii::SolverResults results;
//     if (callback) {
//         double eye_radius = eye.radius;
//         solver.callback_function = [eye_radius, &callback](const spii::CallbackInformation& info) {
//             auto& x = *info.x;

//             Eigen::Matrix<double, 3, 1> eye_pos(x(0), x(1), x(2));
//             Sphere eye(eye_pos, eye_radius);

//             std::vector<Circle> pupils;
//             for (int i = 0; i < (x.size() - 3) / 3; ++i) {
//                 pupils.push_back(EyeModelFitter::circleFromParams(eye, PupilParams(x(3 + 3 * i + 0), x(3 + 3 * i + 1), x(3 + 3 * i + 2))));
//             }

//             callback(eye, pupils);

//             return true;
//         };
//     }
//     solver.solve(f, &results);
//     std::cout << results << std::endl;

//     {
//         std::lock_guard<std::mutex> lock_model(model_mutex);

//         if (current_model_version != model_version)    {
//             std::cout << "Old model, not applying refined parameters" << std::endl;
//             return;
//         }

//         eye.center = x0.segment<3>(0);

//         for (int i = 0; i < (x0.size() - 3) / 3; ++i) {
//             auto pupil_param = x0.segment<3>(3 + 3 * i);
//             pupils[i].params = PupilParams(pupil_param[0], pupil_param[1], pupil_param[2]);
//             pupils[i].circle = circleFromParams(pupils[i].params);
//         }
//     }
// }

void singleeyefitter::EyeModelFitter::initialise_model()
{
    std::lock_guard<std::mutex> lock_model(model_mutex);

    if (eye == Sphere::Null) {
        return;
    }

    // Find pupil positions on eyeball to get radius
    //
    // For each image, calculate the 'most likely' position of the pupil
    // circle given the eyeball sphere estimate and gaze vector. Re-estimate
    // the gaze vector to be consistent with this position.
    // First estimate of pupil center, used only to get an estimate of eye radius
    double eye_radius_acc = 0;
    int eye_radius_count = 0;

    for (const auto& pupil : pupils) {
        if (!pupil.circle) {
            continue;
        }

        if (!pupil.init_valid) {
            continue;
        }

        // Intersect the gaze from the eye center with the pupil circle
        // center projection line (with perfect estimates of gaze, eye
        // center and pupil circle center, these should intersect,
        // otherwise find the nearest point to both lines)
        Vector3 pupil_center = nearest_intersect(Line3(eye.center, pupil.circle.normal),
                               Line3(camera_center, pupil.circle.center.normalized()));
        auto distance = (pupil_center - eye.center).norm();
        eye_radius_acc += distance;
        ++eye_radius_count;
    }

    // Set the eye radius as the mean distance from pupil centers to eye center
    eye.radius = eye_radius_acc / eye_radius_count;

    // Second estimate of pupil radius, used to get position of pupil on eye

    for (auto& pupil : pupils) {
        initialise_single_observation(pupil);
    }

    // Scale eye to anthropomorphic average radius of 12mm
    auto scale = 12.0 / eye.radius;
    eye.radius = 12.0;
    eye.center *= scale;


    for (auto& pupil : pupils) {
        pupil.params.radius *= scale;
        pupil.circle = circleFromParams(pupil.params);
        psi_mean += pupil.params.psi;
        theta_mean += pupil.params.theta;
    }

    psi_mean /= pupils.size();
    theta_mean /= pupils.size();

    model_version++;
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

void singleeyefitter::EyeModelFitter::unproject_observations(double pupil_radius /*= 1*/, double eye_z /*= 20*/, bool use_ransac /*= true*/)
{
    using math::sq;
    std::lock_guard<std::mutex> lock_model(model_mutex);

    if (pupils.size() < 2) {
        return;
    }

    std::vector<std::pair<Circle, Circle>> pupil_unprojection_pairs;
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
        }

    } else {
        // No inliers, so no eye
        eye = Sphere::Null;

        // Arbitrarily pick first circle
        for (size_t i = 0; i < pupils.size(); ++i) {
            const auto& pupil_pair = pupil_unprojection_pairs[i];
            pupils[i].circle = std::move(pupil_pair.first);
        }
    }

    model_version++;
}

void singleeyefitter::EyeModelFitter::fit_circle_for_eye_contours(float max_residual, float max_variance, float min_radius, float max_radius)
{

    if (eye_contours.size() == 0)
        return;

    final_candidate_contours.clear(); // otherwise we fill this infinitly
    // copy the contours
    auto& contours = eye_contours;

    //first we want to filter out the bad stuff, too short ones
    const auto contour_size_min_pred = [](const std::vector<Vector3>& contour) {
        return contour.size() >= 3;
    };
    contours = singleeyefitter::fun::filter(contour_size_min_pred , contours);

    if (contours.size() == 0)
        return;

    // sort the contours so the contour with the most points is at the begining
    std::sort(contours.begin(), contours.end(), [](const std::vector<Vector3>& a, const std::vector<Vector3>& b) { return a.size() < b.size();});

    // saves the best solution and just the Vector3Ds not every single contour
    Contours3D best_solution;
    Circle best_circle;
    double best_variance = std::numeric_limits<double>::infinity();
    double best_goodness = 0.0;
    double best_residual = 0.0;
    float max_fit_residual = max_residual;
    float max_circle_variance = max_variance;

    float pupil_min_radius = min_radius; //approximate min pupil diameter is 2mm in light environment
    float pupil_max_radius = max_radius; //approximate max pupil diameter is 8mm in dark environment


    auto circle_fitter = CircleOnSphereFitter<double>(eye);
    auto circle_evaluation = CircleEvaluation3D<double>(camera_center, eye, max_fit_residual, pupil_min_radius, pupil_max_radius);
    auto circle_variance = CircleDeviationVariance3D<double>();
    auto circle_goodness = CircleGoodness3D<double>();


    auto pruning_quick_combine = [&](const Contours3D & contours,  int max_evals = 1e20, int max_depth = 5) {
        // describes different combinations of contours
        typedef std::set<int> Path;
        // combinations we wanna test
        std::queue<Path> unvisited;

        // contains all the indices for the contours, which altogther fit best
        std::vector<Path> results;

        // contains bad paths, we won't test again
        // even a superset is not tested again, because if a subset is bad, we can't make it better if more contours are added
        std::vector<Path> prune;
        prune.reserve(std::pow(contours.size() , 3));   // we gonna prune a lot if we have alot contours, cubic increase or even more ?
        int eval_count = 0;
        // std::cout << "size:" <<  contours.size()  << std::endl;
        // std::cout << "possible combinations: " <<  std::pow(2,contours.size()) + 1<< std::endl;

        // contains the first moment of each contour
        // we precalculate this inorder to prune contours combinations if the distance of these are to long
        std::vector<Vector3> moments;
        moments.reserve(contours.size());

        // enqueue all contours as starting point
        // and calculate moment
        for (int i = 0; i < contours.size(); i++) {
            unvisited.emplace(std::initializer_list<int> {i});

            Vector3 m = std::accumulate(contours[i].begin(), contours[i].end(), Vector3(0, 0, 0), std::plus<Vector3>());
            m /= contours[i].size();
            moments.push_back(m);
        }

        // inorder to minimize the search space we already prune combinations, which can't fit ,before the search starts
        int prune_count = 0;

        for (int i = 0; i < contours.size(); i++) {
            auto& a = moments[i];

            for (int j = i + 1; j < contours.size(); j++) {
                auto& b = moments[j];
                double distance  = (a - b).squaredNorm();
                double pupil_max_diameter = pupil_max_radius * 2.0;

                if (distance >  std::pow(pupil_max_diameter * 1.3, 2.0)) {
                    prune.emplace_back(std::initializer_list<int> {i, j});
                    prune_count++;
                }
            }
        }

        // std::cout << "pruned " << prune_count << std::endl;

        while (!unvisited.empty() && eval_count <= max_evals) {
            eval_count++;
            //take a path and combine it with others to see if the fit gets better
            Path current_path = unvisited.front();
            unvisited.pop();

            if (current_path.size() <= max_depth) {
                bool includes_bad_paths = fun::isSubset(current_path, prune);

                if (!includes_bad_paths) {
                    int size = 0;

                    for (int j : current_path) { size += contours.at(j).size(); };

                    Contour3D test_contour;

                    Contours3D test_contours;

                    test_contour.reserve(size);

                    std::set<int> test_contour_indices;

                    //concatenate contours to one contour
                    for (int k : current_path) {
                        const Contour3D& c = contours.at(k);
                        test_contours.push_back(c);
                        test_contour.insert(test_contour.end(), c.begin(), c.end());
                        test_contour_indices.insert(k);
                    }

                    //we have not tested this and a subset of this was sucessfull before

                    // need at least 3 points
                    if (!circle_fitter.fit(test_contour)) {
                        std::cout << "Error! Too little points!" << std::endl; // filter too short ones before
                    }

                    // we got a circle fit
                    Circle current_circle = circle_fitter.getCircle();
                    // see if it's even a candidate
                    double variance =  circle_variance(current_circle , test_contours);

                    if (variance <  max_circle_variance) {
                        //yes this was good, keep as solution
                        //results.push_back(test_contour_indices);

                        //lets explore more by creating paths to each remaining node
                        for (int l = (*current_path.rbegin()) + 1 ; l < contours.size(); l++) {
                            unvisited.push(current_path);
                            unvisited.back().insert(l); // add a new path
                        }

                        double residual = circle_fitter.calculateResidual(test_contour);
                        bool is_candidate = circle_evaluation(current_circle, residual);
                        double goodness =  circle_goodness(current_circle , test_contours);

                        if (is_candidate)
                            final_candidate_contours.push_back(test_contours);

                        //check if this one is better then the best one and swap
                        if (is_candidate &&  goodness > best_goodness) {

                            best_residual = residual;
                            best_variance = variance;
                            best_goodness = goodness;
                            best_circle = current_circle;
                            best_solution = test_contours;
                        }

                    } else {
                        prune.push_back(current_path);
                    }
                }
            }
        }

        //std::cout << "tried: "  << eval_count  << std::endl;
        //return results;
    };

    pruning_quick_combine(contours, 1000, 10);

    // std::cout << "residual: " <<  best_residual << std::endl;
    // std::cout << "goodness: " <<  best_goodness << std::endl;
    // std::cout << "variance: " <<  best_variance << std::endl;
    circle_fitted = std::move(best_circle);
    final_circle_contours = std::move(best_solution); // save this for debuging



}


void singleeyefitter::EyeModelFitter::unproject_observation_contours(const Contours_2D& contours)
{
    if (eye == Sphere::Null) {
        return;
    }


    eye_contours.clear();
    eye_contours.resize(contours.size());
    int i = 0;

    for (auto& contour : contours) {
        for (auto& point : contour) {
            Vector3 point_3d(point.x, point.y , focal_length);
            Vector3 direction = point_3d - camera_center;

            try {
                // we use the eye properties of the current eye, when ever we call this
                const auto& unprojected_point = intersect(Line3(camera_center,  direction.normalized()), eye);
                eye_contours[i].push_back(std::move(unprojected_point.first));

            } catch (no_intersection_exception&) {
                // if there is no intersection we don't do anything
            }
        }

        i++;
    }

}
void singleeyefitter::EyeModelFitter::unproject_last_raw_edges()
{
    if (eye == Sphere::Null || pupils.size() == 0) {
        return;
    }


    auto& pupil = pupils.back();
    auto& raw_edges = pupil.observation->raw_edges;
    edges.clear();
    edges.resize(edges.size());
    int i = 0;

    for (auto& point : raw_edges) {
        Vector3 point_3d(point.x, point.y , focal_length);
        Vector3 direction = point_3d - camera_center;

        try {
            // we use the eye properties of the current eye, when ever we call this
            const auto& unprojected_point = intersect(Line3(camera_center,  direction.normalized()), eye);
            edges.push_back(std::move(unprojected_point.first));

        } catch (no_intersection_exception&) {
            // if there is no intersection we don't do anything
        }
    }

}
