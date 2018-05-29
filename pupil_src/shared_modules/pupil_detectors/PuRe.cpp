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

#include "PuRe.h"

#include <climits>
#include <iostream>
//#include <QDebug>
//#include <QElapsedTimer>
#include <opencv2/highgui.hpp>

//#define SAVE_ILLUSTRATION

using namespace std;
using namespace cv;

string PuRe::desc = "PuRe (Santini et. al 2018a)";

PuRe::PuRe() :
	baseSize(320,240),
	expectedFrameSize(-1,-1),
	outlineBias(5)
{
	mDesc = desc;

	/*
	 * 1) Canthi:
	 * Using measurements from white men
	 * Mean intercanthal distance 32.7 (2.4) mm
	 * Mean palpebral fissure width 27.6 (1.9) mm
	 * Jayanth Kunjur, T. Sabesan, V. Ilankovan
	 * Anthropometric analysis of eyebrows and eyelids:
	 * An inter-racial study
	 */
	meanCanthiDistanceMM = 27.6f;
	//meanCanthiDistanceMM = 32.7f;

	/*
	 * 2) Pupil:
	 * 2 to 4 mm in diameter in bright light to 4 to 8 mm in the dark
	 * Clinical Methods: The History, Physical, and Laboratory Examinations. 3rd edition.
	 * Chapter 58The Pupils
	 * Robert H. Spector.
	 */
	maxPupilDiameterMM = 8.0f;
	minPupilDiameterMM = 2.0f;
}

PuRe::~PuRe()
{
}

void PuRe::estimateParameters(int rows, int cols)
{
	/*
	 * Assumptions:
	 * 1) The image contains at least both eye corners
	 * 2) The image contains a maximum of 5cm of the face (i.e., ~= 2x canthi distance)
	 */
	float d = sqrt( pow(rows,2) + pow(cols,2) );
	maxCanthiDistancePx = d;
	minCanthiDistancePx = 2*d/3.0;

	maxPupilDiameterPx = maxCanthiDistancePx*(maxPupilDiameterMM/meanCanthiDistanceMM);
	minPupilDiameterPx = minCanthiDistancePx*(minPupilDiameterMM/meanCanthiDistanceMM);
}

void PuRe::init(const Mat &frame)
{
	if (expectedFrameSize == Size(frame.cols, frame.rows))
		return;

	expectedFrameSize = Size(frame.cols, frame.rows);

	float rw = baseSize.width / (float) frame.cols;
	float rh = baseSize.height / (float) frame.rows;
	scalingRatio = min<float>( min<float>(rw, rh) , 1.0 );
}


Mat PuRe::canny(const Mat &in, bool blurImage, bool useL2, int bins, float nonEdgePixelsRatio, float lowHighThresholdRatio)
{
	(void) useL2;
	/*
	 * Smoothing and directional derivatives
	 * TODO: adapt sizes to image size
	 */
	Mat blurred;
	if (blurImage) {
		Size blurSize(5,5);
		GaussianBlur(in, blurred, blurSize, 1.5, 1.5, BORDER_REPLICATE);
	} else
		blurred = in;

	Sobel(blurred, dx, dx.type(), 1, 0, 7, 1, BORDER_REPLICATE);
	Sobel(blurred, dy, dy.type(), 0, 1, 7, 1, BORDER_REPLICATE);

	/*
	 *  Magnitude
	 */
	double minMag = 0;
	double maxMag = 0;
	float *p_res;
	float *p_x, *p_y; // result, x, y

	cv::magnitude(dx, dy, magnitude);
	cv::minMaxLoc(magnitude, &minMag, &maxMag);

	/*
	 *  Threshold selection based on the magnitude histogram
	 */
	float low_th = 0;
	float high_th = 0;

	// Normalization
	magnitude = magnitude / maxMag;

	// Histogram
	int *histogram = new int[bins]();
	Mat res_idx = (bins-1) * magnitude;
	res_idx.convertTo(res_idx, CV_16U);
	short *p_res_idx=0;
	for(int i=0; i<res_idx.rows; i++){
		p_res_idx = res_idx.ptr<short>(i);
		for(int j=0; j<res_idx.cols; j++)
			histogram[ p_res_idx[j] ]++;
	}

	// Ratio
	int sum=0;
	int nonEdgePixels = nonEdgePixelsRatio * in.rows * in.cols;
	for(int i=0; i<bins; i++){
		sum += histogram[i];
		if( sum > nonEdgePixels ){
			high_th = float(i+1) / bins ;
			break;
		}
	}
	low_th = lowHighThresholdRatio*high_th;

	delete[] histogram;

	/*
	 *  Non maximum supression
	 */
	const float tg22_5 = 0.4142135623730950488016887242097f;
	const float tg67_5 = 2.4142135623730950488016887242097f;
	uchar *_edgeType;
	float *p_res_b, *p_res_t;
	edgeType.setTo(0);
	for(int i=1; i<magnitude.rows-1; i++) {
		_edgeType = edgeType.ptr<uchar>(i);

		p_res=magnitude.ptr<float>(i);
		p_res_t=magnitude.ptr<float>(i-1);
		p_res_b=magnitude.ptr<float>(i+1);

		p_x=dx.ptr<float>(i);
		p_y=dy.ptr<float>(i);

		for(int j=1; j<magnitude.cols-1; j++){

			float m = p_res[j];
			if (m < low_th)
				continue;

			float iy = p_y[j];
			float ix = p_x[j];
			float y  = abs( (double) iy );
			float x  = abs( (double) ix );

			uchar val = p_res[j] > high_th ? 255 : 128;

			float tg22_5x = tg22_5 * x;
			if (y < tg22_5x) {
				if (m > p_res[j-1] && m >= p_res[j+1])
					_edgeType[j] = val;
			} else {
				float tg67_5x = tg67_5 * x;
				if (y > tg67_5x) {
					if (m > p_res_b[j] && m >= p_res_t[j])
						_edgeType[j] = val;
				} else {
					if ( (iy<=0) == (ix<=0) ) {
						if ( m > p_res_t[j-1] && m >= p_res_b[j+1])
							_edgeType[j] = val;
					} else {
						if ( m > p_res_b[j-1] && m >= p_res_t[j+1])
							_edgeType[j] = val;
					}
				}
			}
		}
	}

	/*
	 *  Hystheresis
	 */
	int pic_x=edgeType.cols;
	int pic_y=edgeType.rows;
	int area = pic_x*pic_y;
	int lines_idx=0;
	int idx=0;

	vector<int> lines;
	edge.setTo(0);
	for(int i=1;i<pic_y-1;i++){
		for(int j=1;j<pic_x-1;j++){

			if( edgeType.data[idx+j] != 255 || edge.data[idx+j] != 0 )
				continue;

			edge.data[idx+j] = 255;
			lines_idx = 1;
			lines.clear();
			lines.push_back(idx+j);
			int akt_idx = 0;

			while(akt_idx<lines_idx){
				int akt_pos=lines[akt_idx];
				akt_idx++;

				if( akt_pos-pic_x-1 < 0 || akt_pos+pic_x+1 >= area )
					continue;

				for(int k1=-1;k1<2;k1++)
					for(int k2=-1;k2<2;k2++){
						if(edge.data[(akt_pos+(k1*pic_x))+k2]!=0 || edgeType.data[(akt_pos+(k1*pic_x))+k2]==0)
							continue;
						edge.data[(akt_pos+(k1*pic_x))+k2] = 255;
						lines.push_back((akt_pos+(k1*pic_x))+k2);
						lines_idx++;
					}
			}
		}
		idx+=pic_x;
	}

	return edge;
}


void PuRe::filterEdges(cv::Mat &edges)
{
	// TODO: there is room for improvement here; however, it is prone to small
	// mistakes; will be done when we have time
	int start_x = 5;
	int start_y = 5;
	int end_x = edges.cols - 5;
	int end_y = edges.rows - 5;

	for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){
			uchar box[9];

			box[4]=(uchar)edges.data[(edges.cols*(j))+(i)];

			if(box[4]){
				box[1]=(uchar)edges.data[(edges.cols*(j-1))+(i)];
				box[3]=(uchar)edges.data[(edges.cols*(j))+(i-1)];
				box[5]=(uchar)edges.data[(edges.cols*(j))+(i+1)];
				box[7]=(uchar)edges.data[(edges.cols*(j+1))+(i)];


				if((box[5] && box[7])) edges.data[(edges.cols*(j))+(i)]=0;
				if((box[5] && box[1])) edges.data[(edges.cols*(j))+(i)]=0;
				if((box[3] && box[7])) edges.data[(edges.cols*(j))+(i)]=0;
				if((box[3] && box[1])) edges.data[(edges.cols*(j))+(i)]=0;

			}
		}

	//too many neigbours
	for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){
			uchar neig=0;

			for(int k1=-1;k1<2;k1++)
				for(int k2=-1;k2<2;k2++){

					if(edges.data[(edges.cols*(j+k1))+(i+k2)]>0)
						neig++;
				}

			if(neig>3)
				edges.data[(edges.cols*(j))+(i)]=0;

		}

	for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){
			uchar box[17];

			box[4]=(uchar)edges.data[(edges.cols*(j))+(i)];

			if(box[4]){
				box[0]=(uchar)edges.data[(edges.cols*(j-1))+(i-1)];
				box[1]=(uchar)edges.data[(edges.cols*(j-1))+(i)];
				box[2]=(uchar)edges.data[(edges.cols*(j-1))+(i+1)];

				box[3]=(uchar)edges.data[(edges.cols*(j))+(i-1)];
				box[5]=(uchar)edges.data[(edges.cols*(j))+(i+1)];

				box[6]=(uchar)edges.data[(edges.cols*(j+1))+(i-1)];
				box[7]=(uchar)edges.data[(edges.cols*(j+1))+(i)];
				box[8]=(uchar)edges.data[(edges.cols*(j+1))+(i+1)];

				//external
				box[9]=(uchar)edges.data[(edges.cols*(j))+(i+2)];
				box[10]=(uchar)edges.data[(edges.cols*(j+2))+(i)];


				box[11]=(uchar)edges.data[(edges.cols*(j))+(i+3)];
				box[12]=(uchar)edges.data[(edges.cols*(j-1))+(i+2)];
				box[13]=(uchar)edges.data[(edges.cols*(j+1))+(i+2)];


				box[14]=(uchar)edges.data[(edges.cols*(j+3))+(i)];
				box[15]=(uchar)edges.data[(edges.cols*(j+2))+(i-1)];
				box[16]=(uchar)edges.data[(edges.cols*(j+2))+(i+1)];



				if( (box[10] && !box[7]) && (box[8] || box[6]) ){
					edges.data[(edges.cols*(j+1))+(i-1)]=0;
					edges.data[(edges.cols*(j+1))+(i+1)]=0;
					edges.data[(edges.cols*(j+1))+(i)]=255;
				}


				if( (box[14] && !box[7] && !box[10]) && ( (box[8] || box[6]) && (box[16] || box[15]) ) ){
					edges.data[(edges.cols*(j+1))+(i+1)]=0;
					edges.data[(edges.cols*(j+1))+(i-1)]=0;
					edges.data[(edges.cols*(j+2))+(i+1)]=0;
					edges.data[(edges.cols*(j+2))+(i-1)]=0;
					edges.data[(edges.cols*(j+1))+(i)]=255;
					edges.data[(edges.cols*(j+2))+(i)]=255;
				}



				if( (box[9] && !box[5]) && (box[8] || box[2]) ){
					edges.data[(edges.cols*(j+1))+(i+1)]=0;
					edges.data[(edges.cols*(j-1))+(i+1)]=0;
					edges.data[(edges.cols*(j))+(i+1)]=255;
				}


				if( (box[11] && !box[5] && !box[9]) && ( (box[8] || box[2]) && (box[13] || box[12]) ) ){
					edges.data[(edges.cols*(j+1))+(i+1)]=0;
					edges.data[(edges.cols*(j-1))+(i+1)]=0;
					edges.data[(edges.cols*(j+1))+(i+2)]=0;
					edges.data[(edges.cols*(j-1))+(i+2)]=0;
					edges.data[(edges.cols*(j))+(i+1)]=255;
					edges.data[(edges.cols*(j))+(i+2)]=255;
				}

			}
		}

	for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){

			uchar box[33];

			box[4]=(uchar)edges.data[(edges.cols*(j))+(i)];

			if(box[4]){
				box[0]=(uchar)edges.data[(edges.cols*(j-1))+(i-1)];
				box[1]=(uchar)edges.data[(edges.cols*(j-1))+(i)];
				box[2]=(uchar)edges.data[(edges.cols*(j-1))+(i+1)];

				box[3]=(uchar)edges.data[(edges.cols*(j))+(i-1)];
				box[5]=(uchar)edges.data[(edges.cols*(j))+(i+1)];

				box[6]=(uchar)edges.data[(edges.cols*(j+1))+(i-1)];
				box[7]=(uchar)edges.data[(edges.cols*(j+1))+(i)];
				box[8]=(uchar)edges.data[(edges.cols*(j+1))+(i+1)];

				box[9]=(uchar)edges.data[(edges.cols*(j-1))+(i+2)];
				box[10]=(uchar)edges.data[(edges.cols*(j-1))+(i-2)];
				box[11]=(uchar)edges.data[(edges.cols*(j+1))+(i+2)];
				box[12]=(uchar)edges.data[(edges.cols*(j+1))+(i-2)];

				box[13]=(uchar)edges.data[(edges.cols*(j-2))+(i-1)];
				box[14]=(uchar)edges.data[(edges.cols*(j-2))+(i+1)];
				box[15]=(uchar)edges.data[(edges.cols*(j+2))+(i-1)];
				box[16]=(uchar)edges.data[(edges.cols*(j+2))+(i+1)];

				box[17]=(uchar)edges.data[(edges.cols*(j-3))+(i-1)];
				box[18]=(uchar)edges.data[(edges.cols*(j-3))+(i+1)];
				box[19]=(uchar)edges.data[(edges.cols*(j+3))+(i-1)];
				box[20]=(uchar)edges.data[(edges.cols*(j+3))+(i+1)];

				box[21]=(uchar)edges.data[(edges.cols*(j+1))+(i+3)];
				box[22]=(uchar)edges.data[(edges.cols*(j+1))+(i-3)];
				box[23]=(uchar)edges.data[(edges.cols*(j-1))+(i+3)];
				box[24]=(uchar)edges.data[(edges.cols*(j-1))+(i-3)];

				box[25]=(uchar)edges.data[(edges.cols*(j-2))+(i-2)];
				box[26]=(uchar)edges.data[(edges.cols*(j+2))+(i+2)];
				box[27]=(uchar)edges.data[(edges.cols*(j-2))+(i+2)];
				box[28]=(uchar)edges.data[(edges.cols*(j+2))+(i-2)];

				box[29]=(uchar)edges.data[(edges.cols*(j-3))+(i-3)];
				box[30]=(uchar)edges.data[(edges.cols*(j+3))+(i+3)];
				box[31]=(uchar)edges.data[(edges.cols*(j-3))+(i+3)];
				box[32]=(uchar)edges.data[(edges.cols*(j+3))+(i-3)];

				if( box[7] && box[2] && box[9] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[7] && box[0] && box[10] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[1] && box[8] && box[11] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[1] && box[6] && box[12] )
					edges.data[(edges.cols*(j))+(i)]=0;

				if( box[0] && box[13] && box[17] && box[8] && box[11] && box[21] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[2] && box[14] && box[18] && box[6] && box[12] && box[22] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[6] && box[15] && box[19] && box[2] && box[9] && box[23] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[8] && box[16] && box[20] && box[0] && box[10] && box[24] )
					edges.data[(edges.cols*(j))+(i)]=0;

				if( box[0] && box[25] && box[2] && box[27] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[0] && box[25] && box[6] && box[28] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[8] && box[26] && box[2] && box[27] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[8] && box[26] && box[6] && box[28] )
					edges.data[(edges.cols*(j))+(i)]=0;

				uchar box2[18];
				box2[1]=(uchar)edges.data[(edges.cols*(j))+(i-1)];

				box2[2]=(uchar)edges.data[(edges.cols*(j-1))+(i-2)];
				box2[3]=(uchar)edges.data[(edges.cols*(j-2))+(i-3)];

				box2[4]=(uchar)edges.data[(edges.cols*(j-1))+(i+1)];
				box2[5]=(uchar)edges.data[(edges.cols*(j-2))+(i+2)];

				box2[6]=(uchar)edges.data[(edges.cols*(j+1))+(i-2)];
				box2[7]=(uchar)edges.data[(edges.cols*(j+2))+(i-3)];

				box2[8]=(uchar)edges.data[(edges.cols*(j+1))+(i+1)];
				box2[9]=(uchar)edges.data[(edges.cols*(j+2))+(i+2)];

				box2[10]=(uchar)edges.data[(edges.cols*(j+1))+(i)];

				box2[15]=(uchar)edges.data[(edges.cols*(j-1))+(i-1)];
				box2[16]=(uchar)edges.data[(edges.cols*(j-2))+(i-2)];

				box2[11]=(uchar)edges.data[(edges.cols*(j+2))+(i+1)];
				box2[12]=(uchar)edges.data[(edges.cols*(j+3))+(i+2)];

				box2[13]=(uchar)edges.data[(edges.cols*(j+2))+(i-1)];
				box2[14]=(uchar)edges.data[(edges.cols*(j+3))+(i-2)];

				if( box2[1] && box2[2] && box2[3] && box2[4] && box2[5] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box2[1] && box2[6] && box2[7] && box2[8] && box2[9] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box2[10] && box2[11] && box2[12] && box2[4] && box2[5] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box2[10] && box2[13] && box2[14] && box2[15] && box2[16] )
					edges.data[(edges.cols*(j))+(i)]=0;
			}

		}
}

void PuRe::findPupilEdgeCandidates(const Mat &intensityImage, Mat &edge, vector<PupilCandidate> &candidates)
{
	/* Find all lines
	 * Small note here: using anchor points tends to result in better ellipse fitting later!
	 * It's also faster than doing connected components and collecting the labels
	 */
	vector<Vec4i> hierarchy;
	vector<vector<Point> > curves;
	findContours( edge, curves, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS );

	removeDuplicates(curves, edge.cols);

	// Create valid candidates
	for (size_t i=curves.size(); i-->0;) {
		PupilCandidate candidate(curves[i]);
		if (candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias))
			candidates.push_back( candidate );
	}
}

void PuRe::combineEdgeCandidates(const cv::Mat &intensityImage, cv::Mat &edge, std::vector<PupilCandidate> &candidates)
{
	(void) edge;
	if (candidates.size() <= 1)
		return;
	vector<PupilCandidate> mergedCandidates;
	for (auto pc=candidates.begin(); pc!=candidates.end(); pc++) {
		for (auto pc2=pc+1; pc2!=candidates.end(); pc2++) {

			Rect intersection = pc->combinationRegion & pc2->combinationRegion;
			if (intersection.area() < 1)
				continue; // no intersection
//#define DBG_EDGE_COMBINATION
#ifdef DBG_EDGE_COMBINATION
			Mat tmp;
			cvtColor(intensityImage, tmp, CV_GRAY2BGR);
			rectangle(tmp, pc->combinationRegion, pc->color);
			for (unsigned int i=0; i<pc->points.size(); i++)
				cv::circle(tmp, pc->points[i], 1, pc->color, -1);
			rectangle(tmp, pc2->combinationRegion, pc2->color);
			for (unsigned int i=0; i<pc2->points.size(); i++)
				cv::circle(tmp, pc2->points[i], 1, pc2->color, -1);
			imshow("Combined edges", tmp);
			imwrite("combined.png", tmp);
			//waitKey(0);
#endif

			if (intersection.area() >= min<int>(pc->combinationRegion.area(),pc2->combinationRegion.area()))
				continue;

			vector<Point> mergedPoints = pc->points;
			mergedPoints.insert(mergedPoints.end(), pc2->points.begin(), pc2->points.end());
			PupilCandidate candidate( mergedPoints );
			if (!candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias))
				continue;
			if (candidate.outlineContrast < pc->outlineContrast || candidate.outlineContrast < pc2->outlineContrast)
				continue;
			mergedCandidates.push_back( candidate );
		}
	}
	candidates.insert( candidates.end(), mergedCandidates.begin(), mergedCandidates.end() );
}

void PuRe::searchInnerCandidates(vector<PupilCandidate> &candidates, PupilCandidate &candidate)
{
	if (candidates.size() <= 1)
		return;

	float searchRadius = 0.5*candidate.majorAxis;
	vector<PupilCandidate> insiders;
	for (auto pc=candidates.begin(); pc!=candidates.end(); pc++) {
		if (searchRadius < pc->majorAxis)
			continue;
		if (norm( candidate.outline.center - pc->outline.center) > searchRadius)
			continue;
		if (pc->outlineContrast < 0.75)
			continue;
		insiders.push_back(*pc);
	}
	if (insiders.size() <= 0) {
		//ellipse(dbg, candidate.outline, Scalar(0,255,0));
		return;
	}

	sort( insiders.begin(), insiders.end() );
	candidate = insiders.back();

	//circle(dbg, searchCenter, searchRadius, Scalar(0,0,255),3);
	//candidate.draw(dbg);
	//imshow("dbg", dbg);
}

void PuRe::detect(Pupil &pupil)
{
	// 3.2 Edge Detection and Morphological Transformation
	Mat detectedEdges = canny(input, true, true, 64, 0.7f, 0.4f);

	//imshow("edges", detectedEdges);
#ifdef SAVE_ILLUSTRATION
	imwrite("edges.png", detectedEdges);
#endif
	filterEdges(detectedEdges);

	// 3.3 Segment Selection
	vector<PupilCandidate> candidates;
	findPupilEdgeCandidates(input, detectedEdges, candidates);
	if (candidates.size() <= 0)
		return;

	//for ( auto c = candidates.begin(); c != candidates.end(); c++)
	//	c->draw(dbg);

#ifdef SAVE_ILLUSTRATION
	float r = 255.0 / candidates.size();
	int i = 0;
	Mat candidatesImage;
	cvtColor(input, candidatesImage, CV_GRAY2BGR);
	for ( auto c = candidates.begin(); c != candidates.end(); c++) {
		Mat colorMat = (Mat_<uchar>(1,1) << i*r);
		applyColorMap(colorMat, colorMat, COLORMAP_HSV);
		c->color = colorMat.at<Vec3b>(0,0);
		c->draw(candidatesImage, c->color );
		i++;
	}
	imwrite ("input.png", input);
	imwrite ("filtered-edges.png", detectedEdges);
	imwrite("candidates.png", candidatesImage);
#endif

	// Combination
	combineEdgeCandidates(input, detectedEdges, candidates);
	for (auto c=candidates.begin(); c!=candidates.end(); c++) {
		if (c->outlineContrast < 0.5)
			c->score = 0;
		if (c->outline.size.area() > CV_PI*pow(0.5*maxPupilDiameterPx,2))
			c->score = 0;
		if (c->outline.size.area() < CV_PI*pow(0.5*minPupilDiameterPx,2))
			c->score = 0;
	}

	/*
	for ( int i=0; i<candidates.size(); i++) {
		Mat out;
		cvtColor(input, out, CV_GRAY2BGR);
		auto c = candidates[i];
		c.drawit(out, c.color);
		imwrite(QString("candidate-%1.png").arg(i).toStdString(), out);
		c.drawOutlineContrast(input, 5, QString("contrast-%1-%2.png").arg(i).arg(QString::number(c.score)));
		//waitKey(0);
	}
	*/

	// Scoring
	sort( candidates.begin(), candidates.end() );
	PupilCandidate selected = candidates.back();

	//for ( auto c = candidates.begin(); c != candidates.end(); c++)
	//    c->draw(dbg);

	// Post processing
	searchInnerCandidates(candidates, selected);

	pupil = selected.outline;
	pupil.confidence = selected.outlineContrast;

#ifdef SAVE_ILLUSTRATION
	Mat out;
	cvtColor(input, out, CV_GRAY2BGR);
	ellipse(out, pupil, Scalar(0,255,0), 2);
	line(out, Point(pupil.center.x,0), Point(pupil.center.x,out.rows), Scalar(0,255,0), 2);
	line(out, Point(0,pupil.center.y), Point(out.cols,pupil.center.y), Scalar(0,255,0), 2);
	imwrite("out.png", out);
#endif
}

void PuRe::run(const Mat &frame, Pupil &pupil)
{
	pupil.clear();

	init(frame);

	// Downscaling
	Mat downscaled = frame;
	resize(frame, downscaled, Size(), scalingRatio, scalingRatio, CV_INTER_LINEAR);
	normalize(downscaled, input, 0, 255, NORM_MINMAX, CV_8U);

	workingSize.width = floor(scalingRatio*frame.cols);
	workingSize.height = floor(scalingRatio*frame.rows);

	// Estimate parameters based on the working size
	estimateParameters(workingSize.height, workingSize.width);

	// Preallocate stuff for edge detection
	dx = Mat::zeros(workingSize, CV_32F);
	dy = Mat::zeros(workingSize, CV_32F);
	magnitude = Mat::zeros(workingSize, CV_32F);
	edgeType = Mat::zeros(workingSize, CV_8U);
	edge = Mat::zeros(workingSize, CV_8U);

	//cvtColor(input, dbg, CV_GRAY2BGR);
	//circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*minPupilDiameterPx, Scalar(0,0,0), 2);
	//circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*maxPupilDiameterPx, Scalar(0,0,0), 3);

	// Detection
	detect(pupil);

	pupil.resize( 1.0 / scalingRatio, 1.0 / scalingRatio );

	//imshow("dbg", dbg);
}

void PuRe::run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, const float &userMinPupilDiameterPx, const float &userMaxPupilDiameterPx)
{
	if (roi.area() < 10) {
//		qWarning() << "Bad ROI: falling back to regular detection.";
		run(frame, pupil);
		return;
	}

	pupil.clear();

	init(frame);

	estimateParameters(scalingRatio*frame.rows, scalingRatio*frame.cols);
	if (userMinPupilDiameterPx > 0)
		minPupilDiameterPx = scalingRatio*userMinPupilDiameterPx;
	if (userMaxPupilDiameterPx > 0)
		maxPupilDiameterPx = scalingRatio*userMaxPupilDiameterPx;

	// Downscaling
	Mat downscaled;
	resize(frame(roi), downscaled, Size(), scalingRatio, scalingRatio, CV_INTER_LINEAR);
	normalize(downscaled, input, 0, 255, NORM_MINMAX, CV_8U);

	//cvtColor(input, dbg, CV_GRAY2BGR);

	workingSize.width = input.cols;
	workingSize.height = input.rows;

	// Preallocate stuff for edge detection
	dx = Mat::zeros(workingSize, CV_32F);
	dy = Mat::zeros(workingSize, CV_32F);
	magnitude = Mat::zeros(workingSize, CV_32F);
	edgeType = Mat::zeros(workingSize, CV_8U);
	edge = Mat::zeros(workingSize, CV_8U);

	//cvtColor(input, dbg, CV_GRAY2BGR);
	//circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*minPupilDiameterPx, Scalar(0,0,0), 2);
	//circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*maxPupilDiameterPx, Scalar(0,0,0), 3);

	// Detection
	detect(pupil);

	pupil.resize( 1.0 / scalingRatio, 1.0 / scalingRatio );

	pupil.center += Point2f(roi.tl());
	//imshow("dbg", dbg);
}

/*******************************************************************************
 *
 * Pupil Candidate Functions
 *
 ******************************************************************************/

static const float sinTable[] = {
	0.0000000f  , 0.0174524f  , 0.0348995f  , 0.0523360f  , 0.0697565f  , 0.0871557f  ,
	0.1045285f  , 0.1218693f  , 0.1391731f  , 0.1564345f  , 0.1736482f  , 0.1908090f  ,
	0.2079117f  , 0.2249511f  , 0.2419219f  , 0.2588190f  , 0.2756374f  , 0.2923717f  ,
	0.3090170f  , 0.3255682f  , 0.3420201f  , 0.3583679f  , 0.3746066f  , 0.3907311f  ,
	0.4067366f  , 0.4226183f  , 0.4383711f  , 0.4539905f  , 0.4694716f  , 0.4848096f  ,
	0.5000000f  , 0.5150381f  , 0.5299193f  , 0.5446390f  , 0.5591929f  , 0.5735764f  ,
	0.5877853f  , 0.6018150f  , 0.6156615f  , 0.6293204f  , 0.6427876f  , 0.6560590f  ,
	0.6691306f  , 0.6819984f  , 0.6946584f  , 0.7071068f  , 0.7193398f  , 0.7313537f  ,
	0.7431448f  , 0.7547096f  , 0.7660444f  , 0.7771460f  , 0.7880108f  , 0.7986355f  ,
	0.8090170f  , 0.8191520f  , 0.8290376f  , 0.8386706f  , 0.8480481f  , 0.8571673f  ,
	0.8660254f  , 0.8746197f  , 0.8829476f  , 0.8910065f  , 0.8987940f  , 0.9063078f  ,
	0.9135455f  , 0.9205049f  , 0.9271839f  , 0.9335804f  , 0.9396926f  , 0.9455186f  ,
	0.9510565f  , 0.9563048f  , 0.9612617f  , 0.9659258f  , 0.9702957f  , 0.9743701f  ,
	0.9781476f  , 0.9816272f  , 0.9848078f  , 0.9876883f  , 0.9902681f  , 0.9925462f  ,
	0.9945219f  , 0.9961947f  , 0.9975641f  , 0.9986295f  , 0.9993908f  , 0.9998477f  ,
	1.0000000f  , 0.9998477f  , 0.9993908f  , 0.9986295f  , 0.9975641f  , 0.9961947f  ,
	0.9945219f  , 0.9925462f  , 0.9902681f  , 0.9876883f  , 0.9848078f  , 0.9816272f  ,
	0.9781476f  , 0.9743701f  , 0.9702957f  , 0.9659258f  , 0.9612617f  , 0.9563048f  ,
	0.9510565f  , 0.9455186f  , 0.9396926f  , 0.9335804f  , 0.9271839f  , 0.9205049f  ,
	0.9135455f  , 0.9063078f  , 0.8987940f  , 0.8910065f  , 0.8829476f  , 0.8746197f  ,
	0.8660254f  , 0.8571673f  , 0.8480481f  , 0.8386706f  , 0.8290376f  , 0.8191520f  ,
	0.8090170f  , 0.7986355f  , 0.7880108f  , 0.7771460f  , 0.7660444f  , 0.7547096f  ,
	0.7431448f  , 0.7313537f  , 0.7193398f  , 0.7071068f  , 0.6946584f  , 0.6819984f  ,
	0.6691306f  , 0.6560590f  , 0.6427876f  , 0.6293204f  , 0.6156615f  , 0.6018150f  ,
	0.5877853f  , 0.5735764f  , 0.5591929f  , 0.5446390f  , 0.5299193f  , 0.5150381f  ,
	0.5000000f  , 0.4848096f  , 0.4694716f  , 0.4539905f  , 0.4383711f  , 0.4226183f  ,
	0.4067366f  , 0.3907311f  , 0.3746066f  , 0.3583679f  , 0.3420201f  , 0.3255682f  ,
	0.3090170f  , 0.2923717f  , 0.2756374f  , 0.2588190f  , 0.2419219f  , 0.2249511f  ,
	0.2079117f  , 0.1908090f  , 0.1736482f  , 0.1564345f  , 0.1391731f  , 0.1218693f  ,
	0.1045285f  , 0.0871557f  , 0.0697565f  , 0.0523360f  , 0.0348995f  , 0.0174524f  ,
	0.0000000f  , -0.0174524f , -0.0348995f , -0.0523360f , -0.0697565f , -0.0871557f ,
	-0.1045285f , -0.1218693f , -0.1391731f , -0.1564345f , -0.1736482f , -0.1908090f ,
	-0.2079117f , -0.2249511f , -0.2419219f , -0.2588190f , -0.2756374f , -0.2923717f ,
	-0.3090170f , -0.3255682f , -0.3420201f , -0.3583679f , -0.3746066f , -0.3907311f ,
	-0.4067366f , -0.4226183f , -0.4383711f , -0.4539905f , -0.4694716f , -0.4848096f ,
	-0.5000000f , -0.5150381f , -0.5299193f , -0.5446390f , -0.5591929f , -0.5735764f ,
	-0.5877853f , -0.6018150f , -0.6156615f , -0.6293204f , -0.6427876f , -0.6560590f ,
	-0.6691306f , -0.6819984f , -0.6946584f , -0.7071068f , -0.7193398f , -0.7313537f ,
	-0.7431448f , -0.7547096f , -0.7660444f , -0.7771460f , -0.7880108f , -0.7986355f ,
	-0.8090170f , -0.8191520f , -0.8290376f , -0.8386706f , -0.8480481f , -0.8571673f ,
	-0.8660254f , -0.8746197f , -0.8829476f , -0.8910065f , -0.8987940f , -0.9063078f ,
	-0.9135455f , -0.9205049f , -0.9271839f , -0.9335804f , -0.9396926f , -0.9455186f ,
	-0.9510565f , -0.9563048f , -0.9612617f , -0.9659258f , -0.9702957f , -0.9743701f ,
	-0.9781476f , -0.9816272f , -0.9848078f , -0.9876883f , -0.9902681f , -0.9925462f ,
	-0.9945219f , -0.9961947f , -0.9975641f , -0.9986295f , -0.9993908f , -0.9998477f ,
	-1.0000000f , -0.9998477f , -0.9993908f , -0.9986295f , -0.9975641f , -0.9961947f ,
	-0.9945219f , -0.9925462f , -0.9902681f , -0.9876883f , -0.9848078f , -0.9816272f ,
	-0.9781476f , -0.9743701f , -0.9702957f , -0.9659258f , -0.9612617f , -0.9563048f ,
	-0.9510565f , -0.9455186f , -0.9396926f , -0.9335804f , -0.9271839f , -0.9205049f ,
	-0.9135455f , -0.9063078f , -0.8987940f , -0.8910065f , -0.8829476f , -0.8746197f ,
	-0.8660254f , -0.8571673f , -0.8480481f , -0.8386706f , -0.8290376f , -0.8191520f ,
	-0.8090170f , -0.7986355f , -0.7880108f , -0.7771460f , -0.7660444f , -0.7547096f ,
	-0.7431448f , -0.7313537f , -0.7193398f , -0.7071068f , -0.6946584f , -0.6819984f ,
	-0.6691306f , -0.6560590f , -0.6427876f , -0.6293204f , -0.6156615f , -0.6018150f ,
	-0.5877853f , -0.5735764f , -0.5591929f , -0.5446390f , -0.5299193f , -0.5150381f ,
	-0.5000000f , -0.4848096f , -0.4694716f , -0.4539905f , -0.4383711f , -0.4226183f ,
	-0.4067366f , -0.3907311f , -0.3746066f , -0.3583679f , -0.3420201f , -0.3255682f ,
	-0.3090170f , -0.2923717f , -0.2756374f , -0.2588190f , -0.2419219f , -0.2249511f ,
	-0.2079117f , -0.1908090f , -0.1736482f , -0.1564345f , -0.1391731f , -0.1218693f ,
	-0.1045285f , -0.0871557f , -0.0697565f , -0.0523360f , -0.0348995f , -0.0174524f ,
	-0.0000000f , 0.0174524f  , 0.0348995f  , 0.0523360f  , 0.0697565f  , 0.0871557f  ,
	0.1045285f  , 0.1218693f  , 0.1391731f  , 0.1564345f  , 0.1736482f  , 0.1908090f  ,
	0.2079117f  , 0.2249511f  , 0.2419219f  , 0.2588190f  , 0.2756374f  , 0.2923717f  ,
	0.3090170f  , 0.3255682f  , 0.3420201f  , 0.3583679f  , 0.3746066f  , 0.3907311f  ,
	0.4067366f  , 0.4226183f  , 0.4383711f  , 0.4539905f  , 0.4694716f  , 0.4848096f  ,
	0.5000000f  , 0.5150381f  , 0.5299193f  , 0.5446390f  , 0.5591929f  , 0.5735764f  ,
	0.5877853f  , 0.6018150f  , 0.6156615f  , 0.6293204f  , 0.6427876f  , 0.6560590f  ,
	0.6691306f  , 0.6819984f  , 0.6946584f  , 0.7071068f  , 0.7193398f  , 0.7313537f  ,
	0.7431448f  , 0.7547096f  , 0.7660444f  , 0.7771460f  , 0.7880108f  , 0.7986355f  ,
	0.8090170f  , 0.8191520f  , 0.8290376f  , 0.8386706f  , 0.8480481f  , 0.8571673f  ,
	0.8660254f  , 0.8746197f  , 0.8829476f  , 0.8910065f  , 0.8987940f  , 0.9063078f  ,
	0.9135455f  , 0.9205049f  , 0.9271839f  , 0.9335804f  , 0.9396926f  , 0.9455186f  ,
	0.9510565f  , 0.9563048f  , 0.9612617f  , 0.9659258f  , 0.9702957f  , 0.9743701f  ,
	0.9781476f  , 0.9816272f  , 0.9848078f  , 0.9876883f  , 0.9902681f  , 0.9925462f  ,
	0.9945219f  , 0.9961947f  , 0.9975641f  , 0.9986295f  , 0.9993908f  , 0.9998477f  ,
	1.0000000f
};

static void inline sincos( int angle, float& cosval, float& sinval )
{
	angle += (angle < 0 ? 360 : 0);
	sinval = sinTable[angle];
	cosval = sinTable[450 - angle];
}

static inline vector<Point> ellipse2Points( const RotatedRect &ellipse, const int &delta )
{
	int angle = ellipse.angle;

	// make sure angle is within range
	while( angle < 0 )
		angle += 360;
	while( angle > 360 )
		angle -= 360;

	float alpha, beta;
	sincos( angle, alpha, beta );

	double x, y;
	vector<Point> points;
	for( int i = 0; i < 360; i += delta )
	{
		x = 0.5*ellipse.size.width * sinTable[450-i];
		y = 0.5*ellipse.size.height * sinTable[i];
		points.push_back(
			Point( roundf(ellipse.center.x + x * alpha - y * beta),
				roundf(ellipse.center.y + x * beta + y * alpha) )
			);
	}
	return points;
}

inline bool PupilCandidate::isValid(const cv::Mat &intensityImage, const int &minPupilDiameterPx, const int &maxPupilDiameterPx, const int bias)
{
	if (points.size() < 5)
		return false;

	float maxGap = 0;
	for (auto p1=points.begin(); p1!=points.end(); p1++) {
		for (auto p2=p1+1; p2!=points.end(); p2++) {
			float gap = norm(*p2-*p1);
			if (gap > maxGap)
				maxGap = gap;
		}
	}

	if ( maxGap >= maxPupilDiameterPx )
		return false;
	if ( maxGap <= minPupilDiameterPx )
		return false;

	outline = fitEllipse(points);
	boundaries = {0, 0, intensityImage.cols, intensityImage.rows};

	if (!boundaries.contains(outline.center))
		return false;

	if (!fastValidityCheck(maxPupilDiameterPx) )
		return false;

	pointsMinAreaRect = minAreaRect(points);
	if (ratio(pointsMinAreaRect.size.width,pointsMinAreaRect.size.height) < minCurvatureRatio)
		return false;

	if (!validityCheck(intensityImage, bias))
		return false;

	updateScore();
	return true;
}

inline bool PupilCandidate::fastValidityCheck(const int &maxPupilDiameterPx)
{
	pair<float,float> axis = minmax(outline.size.width, outline.size.height);
	minorAxis = axis.first;
	majorAxis = axis.second;
	aspectRatio = minorAxis / majorAxis;

	if (aspectRatio < minCurvatureRatio)
		return false;

	if (majorAxis > maxPupilDiameterPx)
		return false;

	combinationRegion = boundingRect(points);
	combinationRegion.width = max<int>(combinationRegion.width, combinationRegion.height);
	combinationRegion.height = combinationRegion.width;

	return true;
}

inline bool PupilCandidate::validateOutlineContrast(const Mat &intensityImage, const int &bias)
{
	int delta = 0.15*minorAxis;
	cv::Point c = outline.center;
//#define DBG_OUTLINE_CONTRAST
#ifdef DBG_OUTLINE_CONTRAST
	cv::Mat tmp;
	cv::cvtColor(intensityImage, tmp, CV_GRAY2BGR);
	cv::ellipse(tmp, outline, cv::Scalar(0,255,255));
	cv::Scalar lineColor;
#endif
	int evaluated = 0;
	int validCount = 0;

	vector<Point> outlinePoints = ellipse2Points(outline, 10);
	for (auto p=outlinePoints.begin(); p!=outlinePoints.end(); p++) {
		int dx = p->x - c.x;
		int dy = p->y - c.y;

		float a = 0;
		if (dx != 0)
			a = dy / (float) dx;
		float b = c.y - a*c.x;

		if (a == 0)
			continue;

		if ( abs(dx) > abs(dy) ) {
			int sx = p->x - delta;
			int ex = p->x + delta;
			int sy = std::roundf(a*sx + b);
			int ey = std::roundf(a*ex + b);
			cv::Point start = { sx, sy };
			cv::Point end = { ex, ey };
			evaluated++;

			if (!boundaries.contains(start) || !boundaries.contains(end) )
				continue;

			float m1, m2, count;

			m1 = count = 0;
			for (int x=sx; x<p->x; x++)
				m1 += intensityImage.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int x=p->x+1; x<=ex; x++) {
				m2 += intensityImage.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
			}
			m2 = std::roundf( m2 / delta );

#ifdef DBG_OUTLINE_CONTRAST
			lineColor = cv::Scalar(0,0,255);
#endif
			if (p->x < c.x) {// leftwise point
				if (m1 > m2+bias) {
					validCount ++;
#ifdef DBG_OUTLINE_CONTRAST
					lineColor = cv::Scalar(0,255,0);
#endif
				}
			} else {// rightwise point
				if (m2 > m1+bias) {
					validCount++;
#ifdef DBG_OUTLINE_CONTRAST
					lineColor = cv::Scalar(0,255,0);
#endif
				}
			}

#ifdef DBG_OUTLINE_CONTRAST
			cv::line(tmp, start, end, lineColor);
#endif
		} else {
			int sy = p->y - delta;
			int ey = p->y + delta;
			int sx = std::roundf((sy - b)/a);
			int ex = std::roundf((ey - b)/a);
			cv::Point start = { sx, sy };
			cv::Point end = { ex, ey };

			evaluated++;
			if (!boundaries.contains(start) || !boundaries.contains(end) )
				continue;

			float m1, m2, count;

			m1 = count = 0;
			for (int y=sy; y<p->y; y++)
				m1 += intensityImage.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int y=p->y+1; y<=ey; y++)
				m2 += intensityImage.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
			m2 = std::roundf( m2 / delta );

#ifdef DBG_OUTLINE_CONTRAST
			lineColor = cv::Scalar(0,0,255);
#endif
			if (p->y < c.y) {// upperwise point
				if (m1 > m2+bias) {
					validCount++;
#ifdef DBG_OUTLINE_CONTRAST
					lineColor = cv::Scalar(0,255,0);
#endif
				}
			} else {// bottomwise point
				if (m2 > m1+bias) {
					validCount++;
#ifdef DBG_OUTLINE_CONTRAST
					lineColor = cv::Scalar(0,255,0);
#endif
				}
			}

#ifdef DBG_OUTLINE_CONTRAST
			cv::line(tmp, start, end, lineColor);
#endif
		}
	}
	if (evaluated == 0)
		return false;
	outlineContrast = validCount / (float) evaluated;
#ifdef DBG_OUTLINE_CONTRAST
	cv::imshow("Outline Contrast", tmp);
	cv::waitKey(0);
#endif

	return true;
}

inline bool PupilCandidate::validateAnchorDistribution()
{
	anchorPointSlices.reset();
	for (auto p=points.begin(); p!=points.end(); p++) {
		if (p->x - outline.center.x < 0) {
			if (p->y - outline.center.y < 0)
				anchorPointSlices.set(Q0);
			else
				anchorPointSlices.set(Q3);
		} else  {
			if (p->y - outline.center.y < 0)
				anchorPointSlices.set(Q1);
			else
				anchorPointSlices.set(Q2);
		}
	}
	anchorDistribution = anchorPointSlices.count() / (float) anchorPointSlices.size();
	return true;
}


inline bool PupilCandidate::validityCheck(const cv::Mat &intensityImage, const int &bias)
{
	mp = std::accumulate(points.begin(), points.end(), cv::Point(0,0) );
	mp.x = std::roundf(mp.x / points.size());
	mp.y = std::roundf(mp.y / points.size());

	outline.points(v);
	std::vector<cv::Point2f> pv(v, v+sizeof(v)/sizeof(v[0]));
	if (cv::pointPolygonTest(pv, mp, false) <= 0)
		return false;

	if (!validateAnchorDistribution())
		return false;

	if (!validateOutlineContrast(intensityImage, bias))
		return false;

	return true;
}


//inline bool PupilCandidate::drawOutlineContrast(const Mat &intensityImage, const int &bias, QString out)
//{
//	double lw = 1;
//	int delta = 0.15*minorAxis;
//	cv::Point c = outline.center;
//	cv::Mat tmp;
//	cv::cvtColor(intensityImage, tmp, CV_GRAY2BGR);
//	cv::ellipse(tmp, outline, cv::Scalar(0,255,255), lw);
//	cv::Scalar lineColor;
//
//	int evaluated = 0;
//	int validCount = 0;
//
//
//	vector<Point> outlinePoints = ellipse2Points(outline, 10);
//	for (auto p=outlinePoints.begin(); p!=outlinePoints.end(); p++) {
//		int dx = p->x - c.x;
//		int dy = p->y - c.y;
//
//		float a = 0;
//		if (dx != 0)
//			a = dy / (float) dx;
//		float b = c.y - a*c.x;
//
//		if (a == 0)
//			continue;
//
//		if ( abs(dx) > abs(dy) ) {
//			int sx = p->x - delta;
//			int ex = p->x + delta;
//			int sy = std::roundf(a*sx + b);
//			int ey = std::roundf(a*ex + b);
//			cv::Point start = { sx, sy };
//			cv::Point end = { ex, ey };
//			evaluated++;
//
//			if (!boundaries.contains(start) || !boundaries.contains(end) )
//				continue;
//
//			float m1, m2, count;
//
//			m1 = count = 0;
//			for (int x=sx; x<p->x; x++)
//				m1 += intensityImage.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
//			m1 = std::roundf( m1 / delta );
//
//			m2 = count = 0;
//			for (int x=p->x+1; x<=ex; x++) {
//				m2 += intensityImage.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
//			}
//			m2 = std::roundf( m2 / delta );
//
//			lineColor = cv::Scalar(0,0,255);
//			if (p->x < c.x) {// leftwise point
//				if (m1 > m2+bias) {
//					validCount ++;
//					lineColor = cv::Scalar(0,255,0);
//				}
//			} else {// rightwise point
//				if (m2 > m1+bias) {
//					validCount++;
//					lineColor = cv::Scalar(0,255,0);
//				}
//			}
//
//			cv::line(tmp, start, end, lineColor, lw);
//		} else {
//			int sy = p->y - delta;
//			int ey = p->y + delta;
//			int sx = std::roundf((sy - b)/a);
//			int ex = std::roundf((ey - b)/a);
//			cv::Point start = { sx, sy };
//			cv::Point end = { ex, ey };
//
//			evaluated++;
//			if (!boundaries.contains(start) || !boundaries.contains(end) )
//				continue;
//
//			float m1, m2, count;
//
//			m1 = count = 0;
//			for (int y=sy; y<p->y; y++)
//				m1 += intensityImage.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
//			m1 = std::roundf( m1 / delta );
//
//			m2 = count = 0;
//			for (int y=p->y+1; y<=ey; y++)
//				m2 += intensityImage.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
//			m2 = std::roundf( m2 / delta );
//
//			lineColor = cv::Scalar(0,0,255);
//			if (p->y < c.y) {// upperwise point
//				if (m1 > m2+bias) {
//					validCount++;
//					lineColor = cv::Scalar(0,255,0);
//				}
//			} else {// bottomwise point
//				if (m2 > m1+bias) {
//					validCount++;
//					lineColor = cv::Scalar(0,255,0);
//				}
//			}
//
//			cv::line(tmp, start, end, lineColor, lw);
//		}
//	}
//	if (evaluated == 0)
//		return false;
//	outlineContrast = validCount / (float) evaluated;
//	cv::imwrite(out.toStdString(), tmp);
//
//	return true;
//}
