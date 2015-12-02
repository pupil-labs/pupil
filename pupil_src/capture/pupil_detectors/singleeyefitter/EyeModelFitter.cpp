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
#include "common/constants.h"

#include <Eigen/StdVector>
#include <algorithm>
#include <queue>

namespace singleeyefitter {


    EyeModelFitter::EyeModelFitter(double focalLength, Vector3 cameraCenter) :
        mFocalLength(std::move(focalLength)), mCameraCenter(std::move(cameraCenter)), mCurrentSphere(Sphere::Null), mCurrentInitialSphere(Sphere::Null), mPreviousPupilRadius(0)
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

        for (cv::Point& p : observation2D->raw_edges) {
            p += roi.tl();
            p.x -= image_width_half;
            p.y = image_height_half - p.y;
        }

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
        if (observation2D->confidence  >  0.99) {



            // for (auto& model : mEyeModels) {
            //     model.presentObservation(observation3DPtr);
            // }
            auto circle = mEyeModels.back().presentObservation(observation3DPtr);

            if(circle != Circle::Null)
                mPreviousPupil = circle;



        }else { // if it's too weak we wanna try to find a better one in 3D

            //fitCircle(observation2D->contours, props, result );
            filterCircle(observation2D->raw_edges, props, result );

            if(result.circle != Circle::Null)
                mPreviousPupil = result.circle;
            // project the circle back to 2D
            // need for some calculations in 2D later (calibration)
        }


        if (result.confidence >= 0.9 /*|| observation2D->confidence >= 0.9*/ ) { // either way we get a new circle
            mPreviousPupilRadius = result.circle.radius;
            std::cout << "prev_radius: " << mPreviousPupilRadius << std::endl;
        }
        mCurrentSphere = mEyeModels.back().getSphere();
        std::cout << "current maturity: " << mEyeModels.back().getMaturity() << std::endl;
        mCurrentInitialSphere = mEyeModels.back().getInitialSphere();


        result.sphere = mCurrentSphere;
        result.initialSphere = mCurrentInitialSphere;
        result.binPositions = mEyeModels.back().getBinPositions();
        return result;

    }




    void EyeModelFitter::reset()
    {
        mEyeModels.clear();
        mEyeModels.emplace_back( mFocalLength, mCameraCenter);   // There should be at least one eye-model
        mCurrentSphere = Sphere::Null;
        mCurrentInitialSphere = Sphere::Null;
        mPreviousPupilRadius = 0.0;
    }

    void  EyeModelFitter::fitCircle(const Contours_2D& contours2D , const Detector_3D_Properties& props,  Detector_3D_Result& result) const
    {

        if (contours2D.size() == 0)
            return;

        Contours3D contoursOnSphere  = unprojectObservationContours( contours2D );


        double minRadius = props.pupil_radius_min;
        double maxRadius =  props.pupil_radius_max;

        if (mPreviousPupilRadius != 0.0) {

            minRadius = std::max(mPreviousPupilRadius * 0.85, minRadius );
            maxRadius = std::min(mPreviousPupilRadius * 1.25, maxRadius );
        }
        const double maxDiameter = maxRadius * 2.0;

        //final_candidate_contours.clear(); // otherwise we fill this infinitly

        //first we want to filter out the bad stuff, too short ones
        const auto contour_size_min_pred = [](const std::vector<Vector3>& contour) {
            return contour.size() >= 3;
        };
        contoursOnSphere = singleeyefitter::fun::filter(contour_size_min_pred , contoursOnSphere);

        if (contoursOnSphere.size() == 0)
            return ;

        // sort the contours so the contour with the most points is at the begining
        std::sort(contoursOnSphere.begin(), contoursOnSphere.end(), [](const std::vector<Vector3>& a, const std::vector<Vector3>& b) { return a.size() < b.size();});

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

                            // if (isCandidate)
                            //     final_candidate_contours.push_back(test_contours);

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

        pruning_quick_combine(contoursOnSphere, props.combine_evaluation_max, props.combine_depth_max);

        //std::cout << "residual: " <<  bestResidual << std::endl;
        //std::cout << "goodness: " <<  bestGoodness << std::endl;
        //std::cout << "variance: " <<  bestVariance << std::endl;
        result.circle = std::move(bestCircle);
        result.fittedCircleContours = std::move(bestSolution); // save this for debuging
        result.fitGoodness = bestGoodness;
        result.contours = std::move( contoursOnSphere );
        // project the circle back to 2D
        // need for some calculations in 2D later (calibration)
        result.ellipse = Ellipse(project(bestCircle, mFocalLength));

    }

    Contours3D EyeModelFitter::unprojectObservationContours(const Contours_2D& contours) const
    {
        Contours3D contoursOnSphere;
        contoursOnSphere.resize(contours.size());
        int i = 0;
        //TODO handle contours with no intersection points, because they get closed
        for (auto& contour : contours) {
            for (auto& point : contour) {
                Vector3 point3D(point.x, point.y , mFocalLength);
                Vector3 direction = point3D - mCameraCenter;

                try {
                    // we use the eye properties of the current eye, when ever we call this
                    const auto& unprojectedPoint = intersect(Line3(mCameraCenter,  direction.normalized()), mCurrentSphere);
                    contoursOnSphere[i].push_back(std::move(unprojectedPoint.first));

                } catch (no_intersection_exception&) {
                    // if there is no intersection we don't do anything
                }
            }
            i++;
        }
        return contoursOnSphere;

    }

    Edges3D EyeModelFitter::unprojectEdges(const Edges2D& edges) const
    {
        Edges3D edgesOnSphere;
        edgesOnSphere.reserve(edges.size());
        for (auto& edge : edges) {
            Vector3 point3D(edge.x, edge.y , mFocalLength);
            Vector3 direction = point3D - mCameraCenter;

            try {
                // we use the eye properties of the current eye, when ever we call this
                const auto& unprojectedPoint = intersect(Line3(mCameraCenter,  direction.normalized()), mCurrentSphere);
                edgesOnSphere.push_back(std::move(unprojectedPoint.first));

            } catch (no_intersection_exception&) {
                // if there is no intersection we don't do anything
            }
        }
        return edgesOnSphere;

    }

    void  EyeModelFitter::filterCircle(const Edges2D& rawEdges , const Detector_3D_Properties& props,  Detector_3D_Result& result) const {

        if(rawEdges.size() == 0 )
            return;

        if( mPreviousPupil == Circle::Null)
            return;

        Edges3D edgesOnSphere = unprojectEdges(rawEdges);

        // working just with spherical coords
        std::vector<Vector2> edgesSphericalCoords;
        for( const auto& e : edgesOnSphere){
            Vector3 p  = e - mCurrentSphere.center;
            edgesSphericalCoords.emplace_back( math::cart2sph(p) );
        }

        //const double maxAngularVelocity = constants::PI / 15.0;  // defines the filter space
        const double maxAngularVelocity = 0.2;  // defines the filter space
        const double radiusAngle = std::asin( mPreviousPupil.radius / mCurrentSphere.radius );

        Vector3 c  = mPreviousPupil.center - mCurrentSphere.center;
        Vector2 previousPupilCenter  = math::cart2sph( c );
        double maxTheta = previousPupilCenter.x() + maxAngularVelocity ;
        double maxPsi = previousPupilCenter.y() + maxAngularVelocity ;
        double minTheta = previousPupilCenter.x() - maxAngularVelocity;
        double minPsi = previousPupilCenter.y() - maxAngularVelocity;
        std::cout << "maxtheta: " << maxTheta << std::endl;
        std::cout << "minTheta: " << minTheta << std::endl;

        auto regionFilter = [&]( const Vector2& point ){
            return  point.x() <=  maxTheta + radiusAngle &&
                    point.x() >=  minTheta - radiusAngle&&
                    point.y() <=  maxPsi + radiusAngle &&
                    point.y() >=  minPsi - radiusAngle;
        };

        edgesSphericalCoords = fun::filter( regionFilter, edgesSphericalCoords);

        edgesOnSphere.clear();
        for( const auto& e : edgesSphericalCoords){
            edgesOnSphere.emplace_back(mCurrentSphere.center + math::sph2cart(mCurrentSphere.radius, e.x(), e.y()) );
        }
        result.edges = edgesOnSphere;
        // now we got all edges in the surrounding of the previous pupil, depending on the angular velocity
        // let find the circle where most edges support the circle including a certain region around the circle border

        const double stepSizeAngle = 0.01;
        const double bandWidthAngle = 0.01;
        const double bandWidthAngleHalf = bandWidthAngle/2.0;


        std::cout << "stepsize: " << stepSizeAngle << std::endl;
        std::cout << "bandwidthangle: " << bandWidthAngle << std::endl;
        std::cout << "mintheta: " << minTheta << std::endl;
        std::cout << "maxTheta: " << maxTheta << std::endl;
        std::cout << "minPsi: " << minPsi << std::endl;
        std::cout << "maxPsi: " << maxPsi << std::endl;
        std::cout << "radiusAngle: " << radiusAngle << std::endl;
        int maxEdgeCount = 0;
        Vector2 bestCircleCenter(0,0);

        for (double i = minTheta; i <= maxTheta; i += stepSizeAngle)
        {
              for (double j = minPsi; j <=  maxPsi; j += stepSizeAngle )
              {
                int  edgeCount = 0;
                //count all edges which fall into this current circle
                for( const auto& e : edgesSphericalCoords){

                    //double angelFromCenter = math::haversine(e.x(), e.y(), i , j ); // could we just use pythagoras ? for simpicity
                    // TODO make squared
                    double angelFromCenter = (e - Vector2(i,j)).norm(); // could we just use pythagoras ? for simpicity

                    //std::cout << "angelFromCenter: " << angelFromCenter << std::endl;

                    if( angelFromCenter < radiusAngle+bandWidthAngleHalf  &&
                        angelFromCenter > radiusAngle-bandWidthAngleHalf   ) {

                        edgeCount++;

                    }
                }

                if( edgeCount > maxEdgeCount ){
                    bestCircleCenter = Vector2(i,j);
                    maxEdgeCount = edgeCount;
                }

              }
        }


        if(maxEdgeCount != 0 ){
            result.circle = circleOnSphere( mCurrentSphere, bestCircleCenter.x() , bestCircleCenter.y(), mPreviousPupil.radius );
            std::cout << "edge count: " << maxEdgeCount << std::endl;
            std::cout << "found circle: " << result.circle << std::endl;
        }

    }





} // singleeyefitter
