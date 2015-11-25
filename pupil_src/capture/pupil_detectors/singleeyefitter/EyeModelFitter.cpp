// SingleEyeFitter.cpp : Defines the entry point for the console application.
//


#include "EyeModelFitter.h"
#include "Fit/CircleOnSphereFit.h"
#include "CircleDeviationVariance3D.h"
#include "CircleEvaluation3D.h"
#include "CircleGoodness3D.h"

#include "utils.h"
#include "ImageProcessing/cvx.h"
#include "intersect.h"
#include "projection.h"
#include "fun.h"

#include "mathHelper.h"
#include "distance.h"

#include <Eigen/StdVector>
#include <algorithm>
#include <queue>

namespace singleeyefitter {


    EyeModelFitter::EyeModelFitter(double focalLength, Vector3 cameraCenter) :
        mFocalLength(std::move(focalLength)), mCameraCenter(std::move(cameraCenter)), mCurrentSphere(Sphere::Null), mPreviousPupilRadius(0)
    {
        mEyeModels.emplace_back( mFocalLength, mCameraCenter);   // There should be at least one eye-model
    }


    Detector_3D_Result EyeModelFitter::update_and_detect(std::shared_ptr<Detector_2D_Result>& observation2D , const Detector_3D_Properties& props)
    {

        Detector_3D_Result result;

        // Observations are realtive to their ROI
        cv::Rect roi = observation2D->current_roi;
        int image_height = observation2D->image_height;
        int image_width = observation2D->image_width;
        int image_height_half = image_height / 2.0;
        int image_width_half = image_width / 2.0;

        // For the beginning it's enought to convert just the ellipse
        // If the tests are passed the contours are also converted
        Ellipse& ellipse = observation2D->ellipse;
        ellipse.center[0] -= image_width_half;
        ellipse.center[1] = image_height_half - ellipse.center[1];
        ellipse.angle = -ellipse.angle; //take y axis flip into account

        for (Contour_2D& c : observation2D->contours) {
            for (cv::Point& p : c) {
                p += roi.tl();
                p.x -= image_width_half;
                p.y = image_height_half - p.y;
            }
        }

        // for (Contour_2D& c : observation2D->final_contours) {
        //     for (cv::Point& p : c) {
        //         p += roi.tl();
        //         p.x -= image_width_half;
        //         p.y = image_height_half - p.y;
        //     }
        // }

        // for (cv::Point& p : observation2D->raw_edges) {
        //     p += roi.tl();
        //     p.x -= image_width_half;
        //     p.y = image_height_half - p.y;
        // }

        for (cv::Point& p : observation2D->final_edges) {
            p += roi.tl();
            p.x -= image_width_half;
            p.y = image_height_half - p.y;
        }

        auto observation3DPtr = std::make_shared<const Observation>(observation2D, mFocalLength);

        // Circle unprojected_circle;
        // Circle initialised_circle;

        // if (eye != Sphere::Null){
        //     // let's do this every time, since we need the pupil values anyway
        //     unprojected_circle = selectUnprojectedCircle(pupil ); // unproject circle in 3D space, doesn't consider current eye model (cone unprojection of ellipse)
        //     initialised_circle = initialise_single_observation(pupil);  // initialised circle. circle parameters addapted to our current eye model

        // }

        //check first if the observations is even strong enough to be added
        if (observation2D->confidence  >=  0.9) {



            for (auto& model : mEyeModels) {
                model.presentObservation(observation3DPtr);
            }

            // //if the observation is strong enough, check for other properties if it's a candidate we can use
            // if (eye != Sphere::Null) {

            //     if (unprojected_circle != Circle::Null && initialised_circle != Circle::Null) {  // initialise failed

            //         double support = model_support(unprojected_circle, initialised_circle);
            //         //std::cout << "support: " << support  << std::endl;
            //         if ( support > 0.97  ) {

            //             if (  spatial_variance_check(initialised_circle)) {
            //                 should_add_observation = true;
            //             } else {
            //                 //std::cout << " spatial check failed"  << std::endl;
            //             }

            //         } else {
            //             std::cout << "doesn't support current model "  << std::endl;
            //         }


            //     } else {
            //         std::cout << "no valid circles"  << std::endl;
            //     }

            // } else { // no valid sphere yet
            //     std::cout << "add without check" << std::endl;
            //     should_add_observation = true;
            // }


            // if (should_add_observation) {
            //     //std::cout << "add" << std::endl;

            //     //if the observation passed all tests we can add it
            //     add_observation(Pupil(observation2D));

            //     std::cout << "pupil size " << pupils.size() << std::endl;
            //     //refine model every 50 new pupils
            //     if(pupils.size() > 50 && pupils.size() % 50  == 0){

            //         unproject_observations();
            //         initialise_model();

            //         std::cout << "-----------refine model"  << std::endl;
            //         std::cout << "-----------prev eye: " << eye << std::endl;
            //         refine_with_edges();
            //         std::cout << "-----------new eye: " << eye << std::endl;

            //     }else if(pupils.size() <= 50){

            //         unproject_observations();
            //         initialise_model();
            //     }

            // } else {

            //     // if we don't add a new one we still wanna have the latest pupil parameters
            //     mLatestPupil = std::move(pupil.circle);
            // }

            //std::cout << "2d ellipse " << observation->ellipse << std::endl;


        }// else { // if it's too weak we wanna try to find a better one in 3D

            unproject_observation_contours(observation2D->contours);
            double goodness = fit_circle_for_eye_contours(props);
            result.confidence = goodness;

            // project the circle back to 2D
            // need for some calculations in 2D later (calibration)
            result.ellipse = Ellipse(project(mLatestPupil, mFocalLength));
        // }


        if (result.confidence >= 0.9 || observation2D->confidence >= 0.9) { // either way we get a new circle
            mPreviousPupilRadius = mLatestPupil.radius;
            std::cout << "prev_radius: " << mPreviousPupilRadius << std::endl;
        }


        result.gaze_vector = mLatestPupil.normal; // need to calibrate
        return result;

    }




    void EyeModelFitter::reset()
    {
        // std::lock_guard<std::mutex> lock_model(model_mutex);
        // pupils.clear();
        // pupil_position_bins.clear();
        // bin_positions.clear();
        // eye = Sphere::Null;
        // model_version = 0;
    }

    double EyeModelFitter::fit_circle_for_eye_contours(const Detector_3D_Properties& props)
    {

        if (mContoursOnSphere.size() == 0)
            return 0.0;


        double minRadius = props.pupil_radius_min;
        double maxRadius =  props.pupil_radius_max;

        if (mPreviousPupilRadius != 0.0) {

            minRadius = std::max(mPreviousPupilRadius * 0.85, minRadius );
            maxRadius = std::min(mPreviousPupilRadius * 1.25, maxRadius );
        }
        const double maxDiameter = maxRadius * 2.0;

        final_candidate_contours.clear(); // otherwise we fill this infinitly

        //first we want to filter out the bad stuff, too short ones
        const auto contour_size_min_pred = [](const std::vector<Vector3>& contour) {
            return contour.size() >= 3;
        };
        mContoursOnSphere = singleeyefitter::fun::filter(contour_size_min_pred , mContoursOnSphere);

        if (mContoursOnSphere.size() == 0)
            return 0.0;

        // sort the contours so the contour with the most points is at the begining
        std::sort(mContoursOnSphere.begin(), mContoursOnSphere.end(), [](const std::vector<Vector3>& a, const std::vector<Vector3>& b) { return a.size() < b.size();});

        // saves the best solution and just the Vector3Ds not every single contour
        Contours3D bestSolution;
        Circle bestCircle;
        double bestVariance = std::numeric_limits<double>::infinity();
        double bestGoodness = 0.0;
        double bestResidual = 0.0;

        auto circleFitter = CircleOnSphereFitter<double>(mCurrentSphere);
        auto circleEvaluation = CircleEvaluation3D<double>(mCameraCenter, mCurrentSphere, props.max_fit_residual, minRadius, maxRadius);
        auto circleVariance = CircleDeviationVariance3D<double>();
        auto circleGoodness = CircleGoodness3D<double>();


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
            prune.reserve(std::pow(contours.size() , 3));   // we gonna prune a lot if we have alot contours
            int eval_count = 0;
            //std::cout << "size:" <<  contours.size()  << std::endl;
            //std::cout << "possible combinations: " <<  std::pow(2,contours.size()) + 1<< std::endl;

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
                    double distance_squared  = (a - b).squaredNorm();

                    if (distance_squared >  std::pow(maxDiameter * 1.5, 2.0)) {
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
                        if (!circleFitter.fit(test_contour)) {
                            std::cout << "Error! Too little points!" << std::endl; // filter too short ones before
                        }

                        // we got a circle fit
                        Circle current_circle = circleFitter.getCircle();
                        // see if it's even a candidate
                        double variance =  circleVariance(current_circle , test_contours);

                        if (variance <  props.max_circle_variance) {
                            //yes this was good, keep as solution
                            //results.push_back(test_contour_indices);

                            //lets explore more by creating paths to each remaining node
                            for (int l = (*current_path.rbegin()) + 1 ; l < contours.size(); l++) {
                                // if a new contour is to far away from the current circle center, we can also ignore it
                                // Vector3 contour_moment = moments.at(l);
                                // double distance_squared = (current_circle.center - contour_moment).squaredNorm();
                                // if( distance_squared <   std::pow(pupil_max_radius * 1.5, 2.0) ){
                                //     unvisited.push(current_path);
                                //     unvisited.back().insert(l); // add a new path
                                // }
                                unvisited.push(current_path);
                                unvisited.back().insert(l); // add a new path
                            }

                            double residual = circleFitter.calculateResidual(test_contour);
                            bool isCandidate = circleEvaluation(current_circle, residual);
                            double goodness =  circleGoodness(current_circle , test_contours);

                            if (isCandidate)
                                final_candidate_contours.push_back(test_contours);

                            //check if this one is better then the best one and swap
                            if (isCandidate &&  goodness > bestGoodness) {

                                bestResidual = residual;
                                bestVariance = variance;
                                bestGoodness = goodness;
                                bestCircle = current_circle;
                                bestSolution = test_contours;
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

        pruning_quick_combine(mContoursOnSphere, props.combine_evaluation_max, props.combine_depth_max);

        //std::cout << "residual: " <<  bestResidual << std::endl;
        //std::cout << "goodness: " <<  bestGoodness << std::endl;
        //std::cout << "variance: " <<  bestVariance << std::endl;
        mLatestPupil = std::move(bestCircle);
        final_circle_contours = std::move(bestSolution); // save this for debuging
        return bestGoodness;


    }

    void EyeModelFitter::unproject_observation_contours(const Contours_2D& contours)
    {
        if (mCurrentSphere == Sphere::Null) {
            return;
        }

        mContoursOnSphere.clear();
        mContoursOnSphere.resize(contours.size());
        int i = 0;
        //TODO handle contours with no intersection points, because they get closed
        for (auto& contour : contours) {
            for (auto& point : contour) {
                Vector3 point3D(point.x, point.y , mFocalLength);
                Vector3 direction = point3D - mCameraCenter;

                try {
                    // we use the eye properties of the current eye, when ever we call this
                    const auto& unprojected_point = intersect(Line3(mCameraCenter,  direction.normalized()), mCurrentSphere);
                    mContoursOnSphere[i].push_back(std::move(unprojected_point.first));

                } catch (no_intersection_exception&) {
                    // if there is no intersection we don't do anything
                }
            }
            i++;
        }

    }

} // singleeyefitter
