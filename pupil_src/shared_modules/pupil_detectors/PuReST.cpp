/*
 * Copyright (c) 2018, Thiago Santini
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for non-commercial purposes, without fee, and without a written
 * agreement is hereby granted, provided that:
 *
 * 1) the above copyright notice, this permission notice, and the subsequent
 * bibliographic references be included in all copies or substantial portions of
 * the software
 *
 * 2) the appropriate bibliographic references be made on related publications
 *
 * In this context, non-commercial means not intended for use towards commercial
 * advantage (e.g., as complement to or part of a product) or monetary
 * compensation. The copyright holder reserves the right to decide whether a
 * certain use classifies as commercial or not. For commercial use, please contact
 * the copyright holders.
 *
 * REFERENCES:
 *
 * Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci, PuReST: Robust pupil tracking
 * for real-time pervasive eye tracking, Symposium on Eye Tracking Research and
 * Applications (ETRA), 2018, https://doi.org/10.1145/3204493.3204578.
 *
 * Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci, PuRe: Robust pupil detection
 * for real-time pervasive eye tracking, Computer Vision and Image Understanding,
 * 2018, ISSN 1077-3142, https://doi.org/10.1016/j.cviu.2018.02.002.
 *
 *
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
 * THE AUTHORS HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * THE AUTHORS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE AUTHORS
 * HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS.
*/

#include "PuReST.h"

using namespace std;
using namespace cv;

//#define DBG_BASE_PUPIL
//#define DBG_HIST
//#define DBG_OUTLINE_TRACKER
//#define DBG_GREEDY_TRACKER

std::string PuReST::desc = "PuReST (Santini et al. 2018c)";

void PuReST::calculateHistogram(const cv::Mat &in, cv::Mat &histogram, const int &bins, const Mat &mask)
{
	int channels[] = {0};
	int histSize[] = {bins};
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	calcHist( &in, 1, channels, mask, histogram, 1, histSize, ranges, true, false );
}

static std::vector<cv::Vec3b> getColors(int size)
{
	   std::vector<cv::Vec3b> colors;
	   float r = 255.0 / size;
	   for (int i=0; i<size; i++) {
			   cv::Mat colorMat = (cv::Mat_<uchar>(1,1) << i*r);
			   cv::applyColorMap(colorMat, colorMat, cv::COLORMAP_HSV);
			   colors.push_back( colorMat.at<cv::Vec3b>(0,0) );
	   }
	   return colors;
}

void PuReST::getThresholds(const Mat &input, const Mat &histogram, const Pupil &pupil, int &lowTh, int &highTh, Mat &bright, Mat &dark)
{
	int th;
	float area, acc;

	// High
	acc = 0;
    area = 0.05f * input.rows * input.cols;
	for (th = histogram.rows-1; th > 0; th--) {
		acc += histogram.ptr<float>(th)[0];
		if ( acc > area )
			break;
	}
	highTh = th;

	// Low
	acc = 0;
	area = CV_PI * (0.5 * pupil.size.width) * (0.5 * pupil.size.height);
	for (th = 0; th < histogram.rows; th++) {
		acc += histogram.ptr<float>(th)[0];
		if ( acc > area )
			break;
	}
	lowTh = th;

    int bias = 5;
	highTh -= bias;

	inRange(input, highTh, 256, bright);
	dilate(bright, bright, openKernel);

	inRange(input, 0, lowTh, dark);
	dilate(dark, dark, dilateKernel);
	erode(dark, dark, openKernel);

	//Mat glintCandidates;
	//bitwise_and(bright, dark, glintCandidates);
	//imshow("glints", glintCandidates);


#ifdef DBG_HIST
	float hmax = 0;
	for (int i=0; i<histogram.rows; i++)
		hmax = max<float>(hmax, histogram.ptr<float>(i)[0]);
	Mat v = Mat::zeros(100, histogram.rows, CV_8UC3);
	for (int i=0; i<histogram.rows; i++) {
		double val = histogram.ptr<float>(i)[0] / hmax;
		line(v, Point(i, 0), Point(i, 100*val), Scalar(255,0,0));
	}
	line(v, Point(lowTh, 0), Point(lowTh, 100), Scalar(0,255,0));
	line(v, Point(highTh, 0), Point(highTh, 100), Scalar(0,0,255));
	flip(v, v, 0);
	imshow("Histogram", v);

	Mat dbg;
	cvtColor(input, dbg, CV_GRAY2BGR);
	dbg.setTo( Scalar(0,255,0), dark);
	dbg.setTo( Scalar(0,0,255), bright);
	imshow("dbg", dbg);
#endif

}

void PuReST::generateCombinations(const std::vector<GreedyCandidate> &seeds, std::vector<GreedyCandidate> &candidates, const int length)
{
	if (length > seeds.size())
		return;

	vector<bool> v(seeds.size());
	fill(v.end() - length, v.end(), true);
	do {
		vector<Point> points;
		for (int i = 0; i < seeds.size(); i++) {
			if (v[i]) {
				const vector<Point> &hull = seeds[i].hull;
				points.insert(points.end(), hull.begin(), hull.end());
			}
		}
		candidates.emplace_back( GreedyCandidate(points) );
	} while ( next_permutation(v.begin(), v.end()) );
}

bool PuReST::trackOutline(const cv::Mat &outlineTrackerEdges, const Pupil &basePupil, Pupil &pupil, const float &localScalingRatio, const float &minOutlineConfidence)
{
    vector<Point> edges;

    if (!outlineSeedPupil.valid()) {
        outlineSeedPupil = basePupil;
        outlineSeedPupil.resize( 1.0f / localScalingRatio );
    }

#ifdef DBG_OUTLINE_TRACKER
	Mat dbgOutline;
	cvtColor(input, dbgOutline, CV_GRAY2BGR);
#endif

	// Track previous outline
	float edgeRatio = edgeRatioConfidence(outlineTrackerEdges, basePupil, edges);
#ifdef DBG_OUTLINE_TRACKER
	for ( auto e = edges.begin(); e != edges.end(); e++)
		dbgOutline.at<Vec3b>(e->y, e->x) = Vec3b(0,0,255);
	imshow("Outline Tracker", dbgOutline);
#endif

	Pupil outlineTracker;
	if (edges.size() > 5 && edgeRatio > minOutlineConfidence) {
		outlineTracker = fitEllipse(edges);
		edgeRatio = edgeRatioConfidence(outlineTrackerEdges, outlineTracker, edges);
#ifdef DBG_OUTLINE_TRACKER
		for ( auto e = edges.begin(); e != edges.end(); e++)
			dbgOutline.at<Vec3b>(e->y, e->x) = Vec3b(0,255,255);
		imshow("Outline Tracker", dbgOutline);
#endif
		if (edges.size() > 5 && edgeRatio > minOutlineConfidence) {
			outlineTracker = fitEllipse(edges);
			outlineTracker.confidence = confidence(input, outlineTracker, edges);
#ifdef DBG_OUTLINE_TRACKER
			for ( auto e = edges.begin(); e != edges.end(); e++)
				dbgOutline.at<Vec3b>(e->y, e->x) = Vec3b(0,255,0);
			imshow("Outline Tracker", dbgOutline);
#endif

            // We must compare in the full frame coordinate system because of the dynamic downscaling
            float majorRatio = ( (1.0f / localScalingRatio) * outlineTracker.majorAxis() ) /
                    outlineSeedPupil.majorAxis();

            if ( outlineTracker.valid() && majorRatio < 1.05f ) {
				pupil = outlineTracker;
				return true;
            }
		}
	}

    outlineSeedPupil.clear();

	return false;
}

bool PuReST::greedySearch(const cv::Mat &greedyDetectorEdges, const Pupil &basePupil, const cv::Mat &dark, const cv::Mat &bright, Pupil &pupil, const float &localMinPupilDiameterPx)
{
#ifdef DBG_GREEDY_TRACKER
	Mat dbgGreedy;
	cvtColor(input, dbgGreedy, CV_GRAY2BGR);
#endif

	vector<Vec4i> hierarchy;
	vector<vector<Point> > curves;
    findContours( greedyDetectorEdges, curves, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	for ( auto c = curves.begin(); c != curves.end();) {
		if (c->size() < 5)
			c = curves.erase(c);
		else
			c++;
	}

	// Removes shapes that are too simple
    vector< vector<Point> > approxCurves;
    for ( auto c = curves.begin(); c != curves.end();) {
        vector<Point> ac;
        approxPolyDP( *c, ac, 1.5, false);
        if ( ac.size() > 3 ) {
            approxCurves.push_back(ac);
            c++;
        } else {
            c = curves.erase(c);
        }
    }

	removeDuplicates(curves, greedyDetectorEdges.cols);

	vector<GreedyCandidate> candidates;
	for ( int i = 0; i < curves.size(); i++ ){
		GreedyCandidate c(curves[i]);

        if (c.maxGap > 1.25*basePupil.majorAxis())
			continue;

		float good = 0;
		float regular = 0;
		float bad = 0;
		for (auto p=c.points.begin(); p!=c.points.end(); p++) {
			if ( dark.ptr<uchar>(p->y)[p->x] > 0 ) {
				good++;
			} else {
				if ( bright.ptr<uchar>(p->y)[p->x] > 0 ) {
					bad++;
				} else {
					regular++;
				}
			}
		}

		if (good > bad && good > regular)
			candidates.push_back(std::move(c));
	}

	if (candidates.size() == 0)
		return false;

	// Sort by maxGap
	sort( candidates.begin(), candidates.end(),
		  [](auto &a, auto&b) {
				return a.maxGap > b.maxGap;
			}
		  );

	while (candidates.size() > 5)
		candidates.pop_back();

#ifdef DBG_GREEDY_TRACKER
	vector<Vec3b> colors = getColors(candidates.size());
	for (int i=0; i<candidates.size(); i++) {
		auto &c = candidates[i];
		for ( auto p = c.points.begin(); p != c.points.end(); p++)
			dbgGreedy.at<Vec3b>(p->y, p->x) = colors[i];
	}
	resize(dbgGreedy, dbgGreedy, Size(), 4, 4, INTER_AREA);
	imshow("Greedy Tracker Seeds", dbgGreedy);
	//waitKey(0);
#endif

	vector<GreedyCandidate> combined;
	for (int length = 1; length <= candidates.size(); length++)
		generateCombinations(candidates, combined, length);
	candidates.insert( candidates.end(), make_move_iterator(combined.begin()), make_move_iterator(combined.end()) );

	Pupil greedyPupil;
	float minCurvatureRatio = 0.198912f; // (1-cos(22.5))/sin(22.5)
	for (auto c = candidates.begin(); c != candidates.end(); c++) {
		if (c->hull.size() < 5 )
			continue;
		Pupil p = fitEllipse(c->hull);
		if (p.majorAxis() < localMinPupilDiameterPx)
			continue;
		float aspectRatio = p.minorAxis() / (float) p.majorAxis();
		if ( aspectRatio < minCurvatureRatio)
			continue;
		p.confidence = outlineContrastConfidence(input, p);
		if (p.confidence > greedyPupil.confidence)
			greedyPupil = p;
	}

	if ( greedyPupil.valid(0.66f) ) {
#ifdef DBG_GREEDY_TRACKER
		Mat tmp;
		cvtColor(frame, tmp, CV_GRAY2BGR);
		ellipse(tmp, greedyPupil, Scalar(0,255,0));
		imshow("greedy", tmp);
#endif
		pupil = greedyPupil;
		return true;
	}

	return false;
}
void PuReST::runFromBase(const Timestamp &ts, const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, PupilDetectionMethod &pupilDetectionMethod)
{
    PupilTrackingMethod::run(ts,frame,roi,pupil,pupilDetectionMethod);
}

void PuReST::run(const cv::Mat &frame, const cv::Rect &roi, const Pupil &previousPupil, Pupil &pupil, const float &userMinPupilDiameterPx, const float &userMaxPupilDiameterPx)
{
	(void) roi;
	baseSize = { frame.cols, frame.rows };
	//baseSize = { 320, 240 };
	pupil.clear();

	init(frame);

	// First we get the search region in the frame coordinate system
	Rect frameRect = { 0, 0, frame.cols, frame.rows };
	// TODO: make this dependent on the time difference from previous pupil
	double trackingRectHalfSide = max<int>(previousPupil.size.width, previousPupil.size.height);
	Point2f delta(trackingRectHalfSide, trackingRectHalfSide);
	Rect trackingRect = Rect( previousPupil.center - delta, previousPupil.center + delta);
	trackingRect &= frameRect;

	if (trackingRect.width < 10 || trackingRect.height < 10)
		return;

	float localScalingRatio = scalingRatio;
	Size scaledSize = trackingRect.size();
	scaledSize.width *= scalingRatio;
	scaledSize.height *= scalingRatio;

	// If the resulting rect is too large (e.g., due to a large pupil),
	// we employ a different scale to guarantee runtime
	Size2f maxSize = { 100.f, 100.f };
	if (scaledSize.width > maxSize.width || scaledSize.height > maxSize.height) {
		float r = std::min<float>( maxSize.width / trackingRect.width , maxSize.height / trackingRect.height );
		localScalingRatio = r;
	}

	estimateParameters(localScalingRatio*frame.rows, localScalingRatio*frame.cols);
	if (userMinPupilDiameterPx > 0)
		minPupilDiameterPx = localScalingRatio*userMinPupilDiameterPx;
	if (userMaxPupilDiameterPx > 0)
		maxPupilDiameterPx = localScalingRatio*userMaxPupilDiameterPx;

	/*
	 * From here on, we are in the resulting roi scaled to our base size coordinates
	 */
	resize(frame(trackingRect), input, Size(), localScalingRatio, localScalingRatio, CV_INTER_LINEAR);

	// Setup for Canny
	workingSize = {input.cols, input.rows};
	dx = Mat::zeros(workingSize.height, workingSize.width, CV_32F);
	dy = Mat::zeros(workingSize.height, workingSize.width, CV_32F);
	magnitude = Mat::zeros(workingSize.height, workingSize.width, CV_32F);
	edgeType = Mat::zeros(workingSize.height, workingSize.width, CV_8U);
	edge = Mat::zeros(workingSize.height, workingSize.width, CV_8U);

	// Pupil in our coordinate system
	Pupil basePupil = previousPupil;
	basePupil.shift( -Point2f(trackingRect.tl()) );
	basePupil.resize(localScalingRatio);

#ifdef DBG_BASE_PUPIL
	{
	Mat tmp;
	cvtColor(input, tmp, CV_GRAY2BGR);
	ellipse(tmp, basePupil, Scalar(0,255,0), 2);
	imshow("scaledInput", tmp);
	}
#endif

	// Find glints
	Mat histogram;
	calculateHistogram(input, histogram, 256);

	int lowTh, highTh;
	Mat bright, dark;
	getThresholds(input, histogram, basePupil, lowTh, highTh, bright, dark);

	Mat detectedEdges = canny(input, true, true, 64, 0.7f, 0.4f);
	filterEdges(detectedEdges);

	Mat outlineTrackerEdges = detectedEdges.clone();
	outlineTrackerEdges.setTo(0, bright);
	outlineTrackerEdges.setTo(0, 255 - dark);
	if ( trackOutline(outlineTrackerEdges, basePupil, pupil, localScalingRatio) ) {
		pupil.resize( 1.0 / localScalingRatio );
		pupil.shift( Point2f(trackingRect.tl()) );
		return;
	}

	Mat greedyDetectorEdges = detectedEdges.clone();
    if ( greedySearch(greedyDetectorEdges, basePupil, dark, bright, pupil, localScalingRatio*minPupilDiameterPx) ) {
		pupil.resize( 1.0 / localScalingRatio );
		pupil.shift( Point2f(trackingRect.tl()) );
		return;
    }

    //PuRe::run(frame, roi, pupil, -1, -1);
}

float PuReST::confidence(const cv::Mat frame, const Pupil &pupil, const std::vector<cv::Point> points)
{
	return 0.34*outlineContrastConfidence(frame, pupil) + 0.33*aspectRatioConfidence(pupil) + 0.33*angularSpreadConfidence(points, pupil.center);
}
