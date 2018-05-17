/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#include "common/types.h"
#include "common/colors.h"

#include "singleeyefitter/fun.h"
#include "singleeyefitter/utils.h"
#include "singleeyefitter/ImageProcessing/cvx.h"
#include "geometry/Ellipse.h"  // use ellipse eyefitter
#include "math/distance.h"
#include "singleeyefitter/mathHelper.h"
#include "singleeyefitter/detectorUtils.h"
#include "singleeyefitter/EllipseDistanceApproxCalculator.h"
#include "singleeyefitter/EllipseEvaluation2D.h"
#include "singleeyefitter/ImageProcessing/GuoHallThinner.h"

class Detector2D {

	private:
		typedef singleeyefitter::Ellipse2D<double> Ellipse;


	public:

		Detector2D();
		std::shared_ptr<Detector2DResult> detect(Detector2DProperties& props, cv::Mat& image, cv::Mat& color_image, cv::Mat& debug_image, cv::Rect& roi, bool visualize, bool use_debug_image, bool pause_video);
		std::vector<cv::Point> ellipse_true_support(Detector2DProperties& props, Ellipse& ellipse, double ellipse_circumference, std::vector<cv::Point>& raw_edges);


	private:

		bool mUse_strong_prior;
		int mPupil_Size;
		Ellipse mPrior_ellipse;
		cv::Rect coarsePupilDetection(const cv::Mat &frame, const float &minCoverage, const int &workingWidth, const int &workingHeight);




};

void printPoints(std::vector<cv::Point> points)
{
	std::for_each(points.begin(), points.end(), [](cv::Point & p) { std::cout << p << std::endl;});
}

Detector2D::Detector2D(): mUse_strong_prior(false), mPupil_Size(100) {};

std::vector<cv::Point> Detector2D::ellipse_true_support(Detector2DProperties& props,Ellipse& ellipse, double ellipse_circumference, std::vector<cv::Point>& raw_edges)
{
	std::vector<cv::Point> support_pixels;
	EllipseDistCalculator<double> ellipseDistance(ellipse);

	for (auto& p : raw_edges) {
		double distance = std::abs(ellipseDistance((double)p.x, (double)p.y));
		if (distance <=  props.ellipse_true_support_min_dist) {
			support_pixels.emplace_back(p);
		}
	}
	return support_pixels;
}

cv::Rect Detector2D::coarsePupilDetection(const cv::Mat &frame, const float &minCoverage, const int &workingWidth, const int &workingHeight)
	{
		// We can afford to work on a very small input for haar features, but retain the aspect ratio
		float xr = frame.cols / (float)workingWidth;
		float yr = frame.rows / (float)workingHeight;
		float r = std::min(xr, yr);

		cv::Mat downscaled;
		cv::resize(frame, downscaled, cv::Size(), 1 / r, 1 / r, CV_INTER_LINEAR);

		int ystep = (int)std::max<float>(0.01f*downscaled.rows, 1.0f);
		int xstep = (int)std::max<float>(0.01f*downscaled.cols, 1.0f);

		float d = (float)sqrt(pow(downscaled.rows, 2) + pow(downscaled.cols, 2));

		// Pupil radii is based on PuRe assumptions
		int min_r = (int)(0.5 * 0.05 * d);
		int max_r = (int)(0.5 * 0.12 * d);
		int r_step = (int)std::max<float>(0.1f*(max_r + min_r), 1.0f);

		// TODO: padding so we consider the borders as well!

		cv::Mat itg;
		cv::integral(downscaled, itg, CV_32S);
		cv::Mat res = cv::Mat::zeros(downscaled.rows, downscaled.cols, CV_32F);
		float best_response = std::numeric_limits<float>::min();
		std::deque< std::pair<cv::Rect, float> > candidates;
		for (int r = min_r; r <= max_r; r += r_step) {
			int step = 3 * r;

			cv::Point ia, ib, ic, id;
			cv::Point oa, ob, oc, od;

			int inner_count = (2 * r) * (2 * r);
			int outer_count = (2 * step)*(2 * step) - inner_count;

			float inner_norm = 1.0f / (255 * inner_count);
			float outer_norm = 1.0f / (255 * outer_count);

			for (int y = step; y < downscaled.rows - step; y += ystep) {
				oa.y = y - step;
				ob.y = y - step;
				oc.y = y + step;
				od.y = y + step;
				ia.y = y - r;
				ib.y = y - r;
				ic.y = y + r;
				id.y = y + r;
				for (int x = step; x < downscaled.cols - step; x += xstep) {
					oa.x = x - step;
					ob.x = x + step;
					oc.x = x + step;
					od.x = x - step;
					ia.x = x - r;
					ib.x = x + r;
					ic.x = x + r;
					id.x = x - r;
					int inner = itg.ptr<int>(ic.y)[ic.x] + itg.ptr<int>(ia.y)[ia.x] -
						itg.ptr<int>(ib.y)[ib.x] - itg.ptr<int>(id.y)[id.x];
					int outer = itg.ptr<int>(oc.y)[oc.x] + itg.ptr<int>(oa.y)[oa.x] -
						itg.ptr<int>(ob.y)[ob.x] - itg.ptr<int>(od.y)[od.x] - inner;

					float inner_mean = inner_norm * inner;
					float outer_mean = outer_norm * outer;
					float response = (outer_mean - inner_mean);

					if (response > best_response)
						best_response = response;

					if (response > res.ptr<float>(y)[x]) {
						res.ptr<float>(y)[x] = response;
						// The pupil is too small, the padding too large; we combine them.
						candidates.push_back(std::make_pair(cv::Rect(0.5*(ia + oa), 0.5*(ic + oc)), response));
					}

				}
			}
		}

		// Eliminate low response candidates then sort and combine
		for (auto c = candidates.begin(); c != candidates.end();) {
			if (c->second < 0.5 * best_response)
				c = candidates.erase(c);
			else
				c++;
		}
		auto compare = [](const std::pair<cv::Rect, float> &a, const std::pair<cv::Rect, float> &b) {
			return (a.second > b.second);
		};
		sort(candidates.begin(), candidates.end(), compare);

		// Now add until we reach the minimum coverage or run out of candidates
		cv::Rect coarse;
		int minWidth = minCoverage * downscaled.cols;
		int minHeight = minCoverage * downscaled.rows;
		for (int i = 0; i < candidates.size(); i++) {
			auto &c = candidates[i];
			if (coarse.area() == 0)
				coarse = c.first;
			else
				coarse |= c.first;

			if (coarse.width > minWidth && coarse.height > minHeight)
				break;
		}

		// Upscale result
		coarse.x *= r;
		coarse.y *= r;
		coarse.width *= r;
		coarse.height *= r;

		// Sanity test
		cv::Rect imRoi = cv::Rect(0, 0, frame.cols, frame.rows);
		coarse &= imRoi;
		if (coarse.area() == 0)
			return imRoi;

		return coarse;
	}

std::shared_ptr<Detector2DResult> Detector2D::detect(Detector2DProperties& props, cv::Mat& image, cv::Mat& color_image, cv::Mat& debug_image, cv::Rect& roi, bool visualize, bool use_debug_image, bool pause_video = false)
{
	std::shared_ptr<Detector2DResult> result = std::make_shared<Detector2DResult>();

	roi = this->coarsePupilDetection(cv::Mat(image, roi).clone(), 0.225, 50, 50);

	result->current_roi = roi;
	result->image_width =  image.size().width;
	result->image_height =  image.size().height;

	const int image_width = image.size().width;
	const int image_height = image.size().height;
	const cv::Mat pupil_image = cv::Mat(image, roi).clone();  // image with roi, copy the image, since we alter it
	const int w = pupil_image.size().width / 2;
	const float coarse_pupil_width = w / 2.0f;
	const int padding = int(coarse_pupil_width / 4.0f);
	const int offset = props.intensity_range;
	const int spectral_offset = 5;

	cv::Mat histogram;
	int histSize;
	histSize = 256; //from 0 to 255
	/// Set the ranges
	float range[] = { 0, 256 } ; //the upper boundary is exclusive
	const float* histRange = { range };
	cv::calcHist(&pupil_image, 1 , 0, cv::Mat(), histogram , 1 , &histSize, &histRange, true, false);

	int lowest_spike_index = 255;
	int highest_spike_index = 0;
	float max_intensity = 0;
	singleeyefitter::detector::calculate_spike_indices_and_max_intenesity(histogram, 40, lowest_spike_index, highest_spike_index, max_intensity);

	if (visualize) {
		const int scale_x  = 100;
		const int scale_y = 1 ;

		// display the histogram and the spikes
		for (int i = 0; i < histogram.rows; i++) {
			const float norm_i  = histogram.ptr<float>(i)[0] / max_intensity ; // normalized intensity
			cv::line(color_image, {image_width, i * scale_y}, { image_width - int(norm_i * scale_x), i * scale_y}, mBlue_color);
		}

		cv::line(color_image, {image_width, lowest_spike_index * scale_y}, { int(image_width - 0.5f * scale_x), lowest_spike_index * scale_y }, mRed_color);
		cv::line(color_image, {image_width, (lowest_spike_index + offset)* scale_y}, { int(image_width - 0.5f * scale_x), (lowest_spike_index + offset)* scale_y }, mYellow_color);
		cv::line(color_image, {image_width, (highest_spike_index)* scale_y}, { int(image_width - 0.5f * scale_x), highest_spike_index * scale_y }, mRed_color);
		cv::line(color_image, {image_width, (highest_spike_index - spectral_offset)* scale_y}, { int(image_width - 0.5f * scale_x), (highest_spike_index - spectral_offset)* scale_y }, mWhite_color);
	}

	//create dark and spectral glint masks
	cv::Mat binary_img,spec_mask,kernel;
	cv::inRange(pupil_image, cv::Scalar(0) , cv::Scalar(lowest_spike_index + props.intensity_range), binary_img);    // binary threshold
	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
	cv::dilate(binary_img, binary_img, kernel, { -1, -1}, 2);
	cv::inRange(pupil_image, cv::Scalar(0) , cv::Scalar(highest_spike_index - spectral_offset), spec_mask);    // binary threshold
	cv::erode(spec_mask, spec_mask, kernel);

	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
	//open operation to remove eye lashes
	cv::morphologyEx(pupil_image, pupil_image, cv::MORPH_OPEN, kernel);

	if (props.blur_size > 1)
		cv::medianBlur(pupil_image, pupil_image, props.blur_size);

	cv::Mat edges;
	cv::Canny(pupil_image, edges, props.canny_treshold, props.canny_treshold * props.canny_ration, props.canny_aperture);

	//remove edges in areas not dark enough and where the glint is (spectral refelction from IR leds)
	cv::min(edges, spec_mask, edges);
	cv::min(edges, binary_img, edges);

	if (visualize) {
		// get sub matrix
		cv::Mat overlay = cv::Mat(color_image, roi);
		cv::Mat g_channel(overlay.rows, overlay.cols, CV_8UC1);
		cv::Mat b_channel(overlay.rows, overlay.cols, CV_8UC1);
		cv::Mat r_channel(overlay.rows, overlay.cols, CV_8UC1);
		cv::Mat out[] = {b_channel, g_channel, r_channel};

		cv::split(overlay, out);

		cv::max(g_channel, edges, g_channel);
		cv::max(b_channel, binary_img, b_channel);
		cv::min(b_channel, spec_mask, b_channel);
		cv::merge(out, 3, overlay);

		//draw a frame around the automatic pupil ROI in overlay.
		auto rect = cv::Rect(0, 0, overlay.size().width, overlay.size().height);
		cvx::draw_dotted_rect(overlay, rect, mWhite_color);

		//draw a frame around the area we require the pupil center to be.
		rect = cv::Rect(padding, padding, roi.width - padding, roi.height - padding);
		cvx::draw_dotted_rect(overlay, rect, mWhite_color);

		//draw size ellipses
		cv::Point center(100, image_height - 100);
		cv::circle(color_image, center, props.pupil_size_min / 2.0, mRed_color);

		// real pupil size of this frame is calculated further down, so this size is from the last frame
		cv::circle(color_image, center, mPupil_Size / 2.0, mGreen_color);
		cv::circle(color_image, center, props.pupil_size_max / 2.0, mRed_color);
		auto text_string = std::to_string(mPupil_Size);
		cv::Size text_size = cv::getTextSize(text_string, cv::FONT_HERSHEY_SIMPLEX, 0.4 , 1, 0);
		cv::Point text_pos = { center.x - text_size.width / 2 , center.y + text_size.height / 2};
		cv::putText(color_image, text_string, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.4, mRoyalBlue_color);
	}


	//GuoHallThinner thinner;
    //thinner.thin(edges, true);


	//get raw edge pixel for later
	std::vector<cv::Point> raw_edges;
    // find zero crashes if it doesn't find one. replace with cv implementation if opencv version is 3.0 or above
	singleeyefitter::cvx::findNonZero(edges, raw_edges);


	///////////////////////////////
	/// Strong Prior Part Begin ///
	///////////////////////////////
	//if we had a good ellipse before ,let see if it is still a good first guess:
	if(  mUse_strong_prior ){

	  mUse_strong_prior = false;
	  //recalculate center in coords system of new ROI views! roi changes every framesub
	  Ellipse ellipse = mPrior_ellipse;
	  ellipse.center[0] -= roi.x  ;
	  ellipse.center[1] -= roi.y ;

	  if( !raw_edges.empty() ){

	    std::vector<cv::Point> support_pixels;
	    double ellipse_circumference = ellipse.circumference();
	    support_pixels = ellipse_true_support(props,ellipse, ellipse_circumference, raw_edges);
	    double support_ratio = support_pixels.size() / ellipse_circumference;

	    if(support_ratio >= props.strong_perimeter_ratio_range_min){
	      	cv::RotatedRect refit_ellipse = cv::fitEllipse(support_pixels);

			if(use_debug_image){
			  cv::ellipse(debug_image, toRotatedRect(ellipse), mRoyalBlue_color, 4);
			  cv::ellipse(debug_image, refit_ellipse, mRed_color, 1);
			}

			ellipse = toEllipse<double>(refit_ellipse);

	    	ellipse_circumference = ellipse.circumference();
			auto support_pixels_narrow = ellipse_true_support(props,ellipse, ellipse_circumference, raw_edges);
			props.ellipse_true_support_min_dist *=2.;
			auto support_pixels_wide = ellipse_true_support(props,ellipse, ellipse_circumference, raw_edges);
			props.ellipse_true_support_min_dist /=2.;
	     	support_ratio = pow((float(support_pixels_narrow.size())/float(support_pixels_wide.size())), props.support_pixel_ratio_exponent)  * (float(support_pixels_narrow.size())/ float(ellipse_circumference));
			ellipse.center[0] += roi.x;
			ellipse.center[1] += roi.y;

			mPrior_ellipse = ellipse;
			mUse_strong_prior = true;
			double goodness = std::min(1.0, support_ratio);
			mPupil_Size = ellipse.major_radius * 2.0;
		 	result->confidence = goodness;
			result->ellipse = ellipse;

			//result->final_contours = std::move(best_contours); // no contours when strong prior
			//result->contours = std::move(split_contours);
			result->raw_edges = std::move(raw_edges); // do we need it when strong prior ?
			result->final_edges = std::move(support_pixels);  // need for optimisation
	      	return result;
	    }
	  }
	}
	///////////////////////////////
	///  Strong Prior Part End  ///
	///////////////////////////////

	//from edges to contours
	Contours_2D contours ;
	cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	//first we want to filter out the bad stuff, to short ones
	const auto contour_size_min_pred = [&props](const Contour_2D & contour) {
		return contour.size() > props.contour_size_min;
	};
	contours = singleeyefitter::fun::filter(contour_size_min_pred , contours);

	//now we learn things about each contour through looking at the curvature.
	//For this we need to simplyfy the contour so that pt to pt angles become more meaningfull
	Contours_2D approx_contours;
	std::for_each(contours.begin(), contours.end(), [&](const Contour_2D & contour) {
		std::vector<cv::Point> approx_c;
		cv::approxPolyDP(contour, approx_c, 1.5, false);
		approx_contours.push_back(std::move(approx_c));
	});

	// split contours looking at curvature and angle
	double split_angle = 80;
	int split_contour_size_min = 3;  //removing stubs makes combinatorial search feasable

	//split_contours = singleeyefitter::detector::split_rough_contours(approx_contours, split_angle );
	//removing stubs makes combinatorial search feasable
	//  MOVED TO split_contours_optimized
	//split_contours = singleeyefitter::fun::filter( [](std::vector<cv::Point>& v){ return v.size() <= 3;} , split_contours);
	Contours_2D split_contours = singleeyefitter::detector::split_rough_contours_optimized(approx_contours, split_angle , split_contour_size_min);

	if (split_contours.empty()) {
		result->confidence = 0.0;
		// Does it make seens to return anything ?
		//result->ellipse = toEllipse<double>(refit_ellipse);
		//result->final_contours = std::move(best_contours);
		//result->contours = std::move(split_contours);
		//result->raw_edges = std::move(raw_edges);
		return result;
	}

	if (use_debug_image) {
		// debug segments
		int colorIndex = 0;

		for (const auto& segment : split_contours) {
			const cv::Scalar_<int> colors[] = {mRed_color, mBlue_color, mRoyalBlue_color, mYellow_color, mWhite_color, mGreen_color};
			cv::polylines(debug_image, segment, false, colors[colorIndex], 1, 4);
			colorIndex++;
			colorIndex %= 6;
		}
	}

	std::sort(split_contours.begin(), split_contours.end(), [](const Contour_2D & a,const Contour_2D & b) { return a.size() > b.size(); });
	const cv::Rect ellipse_center_varianz = cv::Rect(padding, padding, pupil_image.size().width - 2.0 * padding, pupil_image.size().height - 2.0 * padding);
	const EllipseEvaluation2D is_Ellipse(ellipse_center_varianz, props.ellipse_roundness_ratio, props.pupil_size_min, props.pupil_size_max);

	//finding potential candidates for ellipse seeds that describe the pupil.
	auto seed_contours  = detector::divide_strong_and_weak_contours(
	                          split_contours, is_Ellipse, props.initial_ellipse_fit_treshhold,
	                          props.strong_perimeter_ratio_range_min, props.strong_perimeter_ratio_range_max,
	                          props.strong_area_ratio_range_min, props.strong_area_ratio_range_max
	                      );
	std::vector<int> seed_indices = seed_contours.first; // strong contours

	if (seed_indices.empty() && !seed_contours.second.empty()) {
		seed_indices = seed_contours.second; // weak contours
	}

	// still empty ? --> exits
	if (seed_indices.empty()) {
		result->confidence = 0.0;
		// Does it make seens to return anything ?
		//result->ellipse = toEllipse<double>(refit_ellipse);
		//result->final_contours = std::move(best_contours);
		//result->contours = std::move(split_contours);
		result->raw_edges = std::move(raw_edges);
		return result;
	}

	auto pruning_quick_combine = [&](const std::vector<std::vector<cv::Point>>& contours,  std::set<int>& seed_indices, int max_evals = 1e20, int max_depth = 5) {
		// describes different combinations of contours
		typedef std::set<int> Path;
		// combinations we wanna test
		std::vector<Path> unknown(seed_indices.size());
		// init with paths of size 1 == seed indices
		int n = 0;
		std::generate(unknown.begin(), unknown.end(), [&]() { return Path{n++}; }); // fill with increasing values, starting from 0
		std::vector<int> mapping; // contains all indices, starting with seed_indices
		mapping.reserve(contours.size());
		mapping.insert(mapping.begin(), seed_indices.begin(), seed_indices.end());

		// add indices which are not used to the end of mapping
		for (int i = 0; i < contours.size(); i++) {
			if (seed_indices.find(i) == seed_indices.end()) { mapping.push_back(i); }
		}

		// contains all the indices for the contours, which altogther fit best
		std::vector<Path> results;

		// contains bad paths, we won't test again
		// even a superset is not tested again, because if a subset is bad, we can't make it better if more contours are added
		std::vector<Path> prune;
		int eval_count = 0;

		while (!unknown.empty() && eval_count <= max_evals) {
			eval_count++;
			//take a path and combine it with others to see if the fit gets better
			Path current_path = unknown.back();
			unknown.pop_back();

			if (current_path.size() <= max_depth) {
				bool includes_bad_paths = fun::isSubset(current_path, prune);

				if (!includes_bad_paths) {
					int size = 0;

					for (int j : current_path) { size += contours.at(mapping.at(j)).size(); };

					std::vector<cv::Point> test_contour;

					test_contour.reserve(size);

					std::set<int> test_contour_indices;

					//concatenate contours to one contour
					for (int k : current_path) {
						const std::vector<cv::Point>& c = contours.at(mapping.at(k));
						test_contour.insert(test_contour.end(), c.begin(), c.end());
						test_contour_indices.insert(mapping.at(k));
					}

					//we have not tested this and a subset of this was sucessfull before
					double fit_variance = detector::contour_ellipse_deviation_variance(test_contour);

					if (fit_variance < props.initial_ellipse_fit_treshhold) {
						//yes this was good, keep as solution
						results.push_back(test_contour_indices);

						//lets explore more by creating paths to each remaining node
						for (int l = (*current_path.rbegin()) + 1 ; l < mapping.size(); l++) {
							unknown.push_back(current_path);
							unknown.back().insert(l); // add a new path
						}

					} else {
						prune.push_back(current_path);
					}
				}
			}
		}

		return results;
	};
	std::set<int> seed_indices_set = std::set<int>(seed_indices.begin(), seed_indices.end());
	std::vector<std::set<int>> solutions = pruning_quick_combine(split_contours, seed_indices_set, 1000, 5);

	//find largest sets which contains all previous ones
	auto filter_subset = [](std::vector<std::set<int>>& sets) {
		std::vector<std::set<int>> filtered_set;
		int i = 0;

		for (auto& current_set : sets) {
			//check if this current_set is a subset of set
			bool isSubset = false;

			for (int j = 0; j < sets.size(); j++) {
				if (j == i) continue;// don't compare to itself

				auto& set = sets.at(j);
				// for std::include both containers need to be ordered. std::set guarantees this
				isSubset |= std::includes(set.begin(), set.end(), current_set.begin(), current_set.end());
			}

			if (!isSubset) {
				filtered_set.push_back(current_set);
			}

			i++;
		}

		return filtered_set;
	};

	solutions = filter_subset(solutions);

    Contours_2D split_contours_resolved(split_contours.size()); // WILL CONTAIN RESOLVED SPLIT CONTOURS TO TEST QUALITY OF CANDIDATE ELLIPSES

    auto resolve_contour = [&](std::vector<cv::Point>& contour, cv::Mat & edges) -> std::vector<cv::Point> {
		cv::Mat support_mask(edges.rows, edges.cols, edges.type(), {0, 0, 0});
		cv::polylines(support_mask, contour, false, {255, 255, 255}, 2);
		cv::Mat new_edges;
		std::vector<cv::Point> new_contours;
		cv::min(edges, support_mask, new_edges);
		cv::findNonZero(new_edges, new_contours);
		return new_contours;
	};

    double max_support_ratio = props.final_perimeter_ratio_range_min; //KEEPS TRACK OF MAXIMUM SUPPORT RATIO REACHED SO FAR

	int index_best_Solution = -1;
	int enum_index = 0;

	for (auto& s : solutions) {

		std::vector<cv::Point> test_contour;

		for (int i : s) {

            //RESOLVE CONTOUR IF IT IS PART OF A SOLUTION AND HAS NOT BEEN RESOLVED YET
		    if (split_contours_resolved[i].size()==0){
                std::vector<cv::Point> generated_edges = resolve_contour(split_contours[i], edges);
                split_contours_resolved[i].reserve(std::distance(generated_edges.begin(), generated_edges.end()));
                split_contours_resolved[i].insert(split_contours_resolved[i].end(), generated_edges.begin(), generated_edges.end());
            }

    		//CONCATENATE CONTOURS TO ONE CONTOUR
			std::vector<cv::Point>& c = split_contours_resolved.at(i);
     		test_contour.insert(test_contour.end(), c.begin(), c.end());

		}

		auto cv_ellipse = cv::fitEllipse(test_contour);

		if (use_debug_image) {
			cv::ellipse(debug_image, cv_ellipse , mRed_color);
		}

		Ellipse ellipse = toEllipse<double>(cv_ellipse);
		double ellipse_circumference = ellipse.circumference();
		std::vector<cv::Point>  support_pixels = ellipse_true_support(props, ellipse, ellipse_circumference, test_contour);
		double support_ratio = (support_pixels.size() / ellipse_circumference)*pow(support_pixels.size()/test_contour.size(), props.support_pixel_ratio_exponent);
		//TODO: refine the selection of final candidate

		if (support_ratio >= max_support_ratio && is_Ellipse(cv_ellipse)) {

			index_best_Solution = enum_index;
            max_support_ratio = support_ratio;

			if (support_ratio >= props.strong_perimeter_ratio_range_min) {
				ellipse.center[0] += roi.x;
				ellipse.center[1] += roi.y;
				mPrior_ellipse = ellipse;
				mUse_strong_prior = true;

				if (use_debug_image) {
					cv::ellipse(debug_image, cv_ellipse , mGreen_color);
				}
			}
		}

		enum_index++;
	}

	// select ellipse

	if (index_best_Solution == -1) {
		// no good final ellipse found
		result->confidence = 0.0;
		// Does it make seens to return anything ?
		//result->ellipse = toEllipse<double>(refit_ellipse);
		//result->final_contours = std::move(best_contours);
		//result->contours = std::move(split_contours);
		result->raw_edges = std::move(raw_edges);
		return result;
	}

	auto& best_solution = solutions.at(index_best_Solution);
	std::vector<std::vector<cv::Point>> best_contours;
	std::vector<cv::Point>best_contour;

	//concatenate contours to one contour
	for (int i : best_solution) {
		std::vector<cv::Point>& c = split_contours.at(i);
		best_contours.push_back(c);
		best_contour.insert(best_contour.end(), c.begin(), c.end());
	}

	auto cv_ellipse = cv::fitEllipse(best_contour);
	//final fitting on resolved contour
	auto final_fitting = [&](std::vector<std::vector<cv::Point>>& contours, cv::Mat & edges) -> std::vector<cv::Point> {
		//use the real edge pixels to fit, not the aproximated contours
		cv::Mat support_mask(edges.rows, edges.cols, edges.type(), {0, 0, 0});
		cv::polylines(support_mask, contours, false, {255, 255, 255}, 2);

		//draw into the suport mask with thickness 2
		cv::Mat new_edges;
		std::vector<cv::Point> new_contours;
		cv::min(edges, support_mask, new_edges);

		// can't do this here, because final result gets much distorted.
		// see if it even can crash !!!
		//new_edges.at<int>(0,0) = 1; // find zero crashes if it doesn't find one. remove if opencv version is 3.0 or above
		cv::findNonZero(new_edges, new_contours);

		if (visualize)
		{
			cv::Mat overlay = color_image.colRange(roi.x, roi.x + roi.width).rowRange(roi.y, roi.y + roi.height);
			cv::Mat g_channel(overlay.rows, overlay.cols, CV_8UC1);
			cv::Mat b_channel(overlay.rows, overlay.cols, CV_8UC1);
			cv::Mat r_channel(overlay.rows, overlay.cols, CV_8UC1);
			cv::Mat out[] = {b_channel, g_channel, r_channel};
			cv::split(overlay, out);
			cv::threshold(new_edges, new_edges, 0, 255, cv::THRESH_BINARY);
			cv::max(r_channel, new_edges, r_channel);
			cv::merge(out, 3, overlay);
		}

		return new_contours;

	};
	std::vector<cv::Point> final_edges = final_fitting(best_contours, edges);
	auto cv_new_Ellipse = cv::fitEllipse(final_edges);
	double size_difference  = std::abs(1.0 - cv_ellipse.size.height / cv_new_Ellipse.size.height);
	auto& cv_final_Ellipse = cv_ellipse;
	if (is_Ellipse(cv_new_Ellipse) && size_difference < 0.3) {
		if (use_debug_image) {
			cv::ellipse(debug_image, cv_new_Ellipse, mGreen_color);
		}
		cv_final_Ellipse = cv_new_Ellipse;
	}

	//cv::imshow("debug_image", debug_image);
	mPupil_Size =  cv_final_Ellipse.size.height;
	result->ellipse = toEllipse<double>(cv_final_Ellipse);

    // final calculation of goodness of fit on final_edges
    double ellipse_circumference = (result->ellipse).circumference();
    std::vector<cv::Point>  support_pixels = ellipse_true_support(props, result->ellipse, ellipse_circumference, final_edges);
	double support_ratio = support_pixels.size() / ellipse_circumference;
	double goodness = std::min(double(0.99),support_ratio)*pow(support_pixels.size()/final_edges.size(), props.support_pixel_ratio_exponent);

	result->confidence = goodness;
	result->ellipse.center[0] += roi.x;
	result->ellipse.center[1] += roi.y;
	//result->final_contours = std::move(best_contours);

	// TODO optimize
	// just do this if we really need it
	// std::for_each(contours.begin(), contours.end(), [&](const Contour_2D & contour) {
	// 	std::vector<cv::Point> approx_c;
	// 	cv::approxPolyDP(contour, approx_c, 1.0, false);
	// 	approx_contours.push_back(std::move(approx_c));
	// });
	// split_contours = singleeyefitter::detector::split_rough_contours_optimized(approx_contours, 150.0 , split_contour_size_min);

	// result->contours = std::move(split_contours);
	result->final_edges = std::move(final_edges);// need for optimisation

	result->raw_edges = std::move(raw_edges);
	return result;

}

