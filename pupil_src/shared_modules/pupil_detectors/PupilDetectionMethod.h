#ifndef PUPILDETECTIONMETHOD_H
#define PUPILDETECTIONMETHOD_H

//#include <QMetaType>
//#include <QDebug>

#include <string>
#include <deque>
#include <bitset>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define NO_CONFIDENCE -1.0
#define SMALLER_THAN_NO_CONFIDENCE NO_CONFIDENCE-1.0

class Pupil : public cv::RotatedRect {
public:
	Pupil(const RotatedRect &outline, const float &confidence) :
		RotatedRect(outline),
		confidence(confidence) {}

	Pupil(const RotatedRect &outline) :
		RotatedRect(outline),
		confidence(NO_CONFIDENCE) { }

	Pupil() { clear(); }

	float confidence;

	void clear() {
		angle = -1.0;
		center = { -1.0, -1.0 };
		size = { -1.0, -1.0 };
		confidence = NO_CONFIDENCE;
    }

	void resize(const float &xf, const float &yf) {
		center.x *= xf;
		center.y *= yf;
		size.width *= xf;
		size.height *= yf;
	}
	void resize(const float &f) {
		center *= f;
		size *= f;
	}
	void shift( cv::Point2f p ) { center += p; }

	bool valid(const double &confidenceThreshold=SMALLER_THAN_NO_CONFIDENCE) const {
		return center.x > 0 &&
			center.y > 0 &&
			size.width > 0 &&
			size.height > 0 &&
			confidence > confidenceThreshold;
	}

	bool hasOutline() const { return size.width > 0 && size.height > 0; }
	int majorAxis() const { return std::max<int>(size.width, size.height); }
	int minorAxis() const { return std::min<int>(size.width, size.height); }
	int diameter() const { return majorAxis(); }
	float circumference() const {
		float a = 0.5*majorAxis();
		float b = 0.5*minorAxis();
		return CV_PI * abs( 3*(a+b) - sqrt( 10*a*b + 3*( pow(a,2) + pow(b,2) ) ) );
	}
};

//Q_DECLARE_METATYPE(Pupil);

class PupilDetectionMethod
{
public:
    PupilDetectionMethod() {}
    ~PupilDetectionMethod() {}

    virtual cv::RotatedRect run(const cv::Mat &frame) = 0;
	virtual bool hasConfidence() = 0;
	virtual bool hasCoarseLocation() = 0;
	std::string description() { return mDesc; }

	virtual void run(const cv::Mat &frame, Pupil &pupil) {
        pupil.clear();
		pupil = run(frame);
		pupil.confidence = 1;
    }
	virtual void run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, const float &minPupilDiameterPx=-1, const float &maxPupilDiameterPx=-1) {
		(void) roi;
		(void) minPupilDiameterPx;
		(void) maxPupilDiameterPx;
		run(frame, pupil);
	}

	// Pupil detection interface used in the tracking
	Pupil runWithConfidence(const cv::Mat &frame, const cv::Rect &roi, const float &minPupilDiameterPx=-1, const float &maxPupilDiameterPx=-1) {
		Pupil pupil;
		run(frame, roi, pupil, minPupilDiameterPx, maxPupilDiameterPx);
		if ( ! hasConfidence() )
			pupil.confidence = outlineContrastConfidence(frame, pupil);
		return pupil;
	}

	virtual Pupil getNextCandidate() { return Pupil(); }

	// Generic coarse pupil detection
	static cv::Rect coarsePupilDetection(const cv::Mat &frame, const float &minCoverage=0.5f, const int &workingWidth=60, const int &workingHeight=40);

	// Generic confidence metrics
	static float outlineContrastConfidence(const cv::Mat &frame, const Pupil &pupil, const int &bias=5);
	static float edgeRatioConfidence(const cv::Mat &edgeImage, const Pupil &pupil, std::vector<cv::Point> &edgePoints, const int &band=5);
	static float angularSpreadConfidence(const std::vector<cv::Point> &points, const cv::Point2f &center);
	static float aspectRatioConfidence(const Pupil &pupil);

	//Pupil test(const cv::Mat &frame, const cv::Rect &roi, Pupil pupil) { return pupil; }
protected:
	std::string mDesc;
};


#endif // PUPILDETECTIONMETHOD_H
