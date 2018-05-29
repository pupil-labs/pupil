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

#ifndef PURE_H
#define PURE_H

#include <bitset>
#include <random>

//#include <QString>
//#include <QDebug>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "PupilDetectionMethod.h"

class PupilCandidate
{
public:
    std::vector<cv::Point> points;
    cv::RotatedRect pointsMinAreaRect;
    float minCurvatureRatio;

    cv::RotatedRect outline;

    cv::Rect pointsBoundingBox;
    cv::Rect combinationRegion;
    cv::Rect br;
    cv::Rect boundaries;
    cv::Point2f v[4];
    cv::Rect outlineInscribedRect;
    cv::Point2f mp;
    float minorAxis, majorAxis;
    float aspectRatio;
    cv::Mat internalArea;
    float innerMeanIntensity;
    float outerMeanIntensity;
    float contrast;
    float outlineContrast;
    float anchorDistribution;
    float score;
	std::bitset<4> anchorPointSlices;

	cv::Scalar color;

    enum {
        Q0 = 0,
        Q1 = 1,
        Q2 = 2,
        Q3 = 3,
    };

	PupilCandidate(std::vector<cv::Point> points) :
        minCurvatureRatio(0.198912f), // (1-cos(22.5))/sin(22.5)
        anchorDistribution(0.0f),
        aspectRatio(0.0f),
        outlineContrast(0.0f),
		score(0.0f),
		color(0,255,0)
    {
        this->points = points;
    }
    bool isValid(const cv::Mat &intensityImage, const int &minPupilDiameterPx, const int &maxPupilDiameterPx, const int bias=5);
    void estimateOutline();
    bool isCurvatureValid();

    // Support functions
    float ratio(float a, float b) {
        std::pair<float,float> sorted = std::minmax(a,b);
        return sorted.first / sorted.second;
    }

    bool operator < (const PupilCandidate& c) const
    {
        return (score < c.score);
    }

    bool fastValidityCheck(const int &maxPupilDiameterPx);

    bool validateAnchorDistribution();

    bool validityCheck(const cv::Mat &intensityImage, const int &bias);

	bool validateOutlineContrast(const cv::Mat &intensityImage, const int &bias);
//	bool drawOutlineContrast(const cv::Mat &intensityImage, const int &bias, QString out);

    void updateScore()
    {
        score = 0.33*aspectRatio + 0.33*anchorDistribution + 0.34*outlineContrast;
        // ElSe style
        //score = (1-innerMeanIntensity)*(1+abs(outline.size.height-outline.size.width));
    }

    void draw(cv::Mat out){
        //cv::ellipse(out, outline, cv::Scalar(0,255,0));
        //cv::rectangle(out, combinationRegion, cv::Scalar(0,255,255));
        for (unsigned int i=0; i<points.size(); i++)
            cv::circle(out, points[i], 1, cv::Scalar(0, 255, 255));

        cv::circle(out, mp, 3, cv::Scalar(0, 0, 255), -1);
//        QString s = QString("%1 %2 %3").arg(
//                        QString::number(aspectRatio,'g', 2)
//                        ).arg(
//                        QString::number(anchorDistribution,'g', 2)
//                        ).arg(
//                        QString::number(contrast,'g', 2));
        //cv::putText(out, s.toStdString(), outline.center, CV_FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255));
        //cv::putText(out, QString::number(score,'g', 2).toStdString(), outline.center, CV_FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255));
        //cv::putText(out, QString::number(anchorDistribution,'g', 2).toStdString(), outline.center, CV_FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255));
    }

    void draw(cv::Mat out, cv::Scalar color){
		int w = 2;
        cv::circle(out, points[0], w, color, -1);
        for (unsigned int i=1; i<points.size(); i++) {
            cv::circle(out, points[i], w, color, -1);
            cv::line(out, points[i-1], points[i], color, w-1);
        }
        cv::line(out, points[points.size()-1], points[0], color, w-1);
	}

	void drawit(cv::Mat out, cv::Scalar color){
		int w = 2;
		for (unsigned int i=0; i<points.size(); i++)
			cv::circle(out, points[i], w, color, -1);
		cv::ellipse(out, outline, color);
	}

};

class PuRe : public PupilDetectionMethod
{
public:
    PuRe();
    ~PuRe();

    cv::RotatedRect run(const cv::Mat &frame) {
        Pupil pupil;
        run(frame, pupil);
		return pupil;
    }

    void run(const cv::Mat &frame, Pupil &pupil);
	void run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, const float &userMinPupilDiameterPx=-1, const float &userMaxPupilDiameterPx=-1);
	bool hasPupilOutline() { return true; }
	bool hasConfidence() { return true; }
	bool hasCoarseLocation() { return false; }
	static std::string desc;

    float meanCanthiDistanceMM;
    float maxPupilDiameterMM;
    float minPupilDiameterMM;
    float meanIrisDiameterMM;

protected:
    cv::RotatedRect detectedPupil;
    cv::Size expectedFrameSize;

    int outlineBias;

    static const cv::RotatedRect invalidPupil;

    /*
     *  Initialization
     */
    void init(const cv::Mat &frame);
    void estimateParameters(int rows, int cols);

    /*
     * Downscaling
     */
    cv::Size baseSize;
    cv::Size workingSize;
	float scalingRatio;

    /*
     *  Detection
     */
    void detect(Pupil &pupil);

    // Canny
	cv::Mat dx, dy, magnitude;
    cv::Mat edgeType, edge;
	cv::Mat canny(const cv::Mat &in, bool blur=true, bool useL2=true, int bins=64, float nonEdgePixelsRatio=0.7f, float lowHighThresholdRatio=0.4f);

    // Edge filtering
	void filterEdges(cv::Mat &edges);

	// Remove duplicates (e.g., from closed loops)
	int pointHash(cv::Point p, int cols) { return p.y*cols+p.x;}
	void removeDuplicates(std::vector<std::vector<cv::Point> > &curves, const int& cols) {
		std::map<int,uchar> contourMap;
		for (size_t i=curves.size(); i-->0;) {
			if (contourMap.count(pointHash(curves[i][0],cols)) > 0)
				curves.erase(curves.begin()+i);
			else {
				for (int j=0; j<curves[i].size(); j++)
					contourMap[pointHash(curves[i][j],cols)] = 1;
			}
		}
	}

    void findPupilEdgeCandidates(const cv::Mat &intensityImage, cv::Mat &edge, std::vector<PupilCandidate> &candidates);
    void combineEdgeCandidates(const cv::Mat &intensityImage, cv::Mat &edge, std::vector<PupilCandidate> &candidates);
	void searchInnerCandidates(std::vector<PupilCandidate> &candidates, PupilCandidate &candidate);

    cv::Mat input;
    cv::Mat dbg;

    int maxCanthiDistancePx;
    int minCanthiDistancePx;
    int maxPupilDiameterPx;
    int minPupilDiameterPx;

};

#endif // PURE_H
