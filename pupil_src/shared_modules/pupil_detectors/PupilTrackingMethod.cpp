#include "PupilTrackingMethod.h"

//#include <QtConcurrent/QtConcurrent>
//#include <QFuture>

using namespace std;
using namespace cv;

void PupilTrackingMethod::reset()
{
	previousPupils.clear();
	previousPupil = TrackedPupil();
	pupilDiameterKf.statePost.ptr<float>(0)[0] = 0.5*expectedFrameSize.width;
}

void PupilTrackingMethod::registerPupil( const Timestamp &ts, Pupil &pupil ) {
	Mat measurement = ( Mat_<float>(1,1) << pupil.majorAxis() );
	//if (predictedMaxPupilDiameter > 0) {
	//	float &majorAxis = measurement.ptr<float>(0)[0];
	//	if ( majorAxis > predictedMaxPupilDiameter) {
	//		pupil.clear();
	//		return;
	//	}
	//}

	if (pupil.confidence > minDetectionConfidence) {
		previousPupil = TrackedPupil(ts, pupil);
		previousPupils.emplace_back( previousPupil );
		pupilDiameterKf.correct(measurement);
	} else
		previousPupil = TrackedPupil();

	//if (pupil.confidence > minDetectionConfidence) {
	//	previousPupil = TrackedPupil(ts, pupil);
	//	previousPupils.push_back( previousPupil );
	//} else
	//	previousPupil = TrackedPupil();
}

void PupilTrackingMethod::predictMaxPupilDiameter() {
	predictedMaxPupilDiameter = 1.5*pupilDiameterKf.predict().ptr<float>(0)[0];
	if (previousPupils.size() < 20)
		predictedMaxPupilDiameter = -1;
}

void PupilTrackingMethod::run(const Timestamp &ts, const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, PupilDetectionMethod &pupilDetectionMethod)
{
	cv::Size frameSize = { frame.cols, frame.rows };
	if (expectedFrameSize != frameSize ) {
		// Reference frame changed. Let's start over!
		expectedFrameSize = frameSize;
		reset();
	}

	// Remove old samples
	while (!previousPupils.empty()) {
		if (ts - previousPupils.front().ts > maxAge)
			previousPupils.pop_front();
		else
			break;
	}

	pupil.clear();
	predictMaxPupilDiameter();

	if ( previousPupil.confidence == NO_CONFIDENCE ) {
		pupil = pupilDetectionMethod.runWithConfidence(frame, roi, -1, -1);
	} else {
		run(frame, roi, previousPupil, pupil);
	}

	registerPupil(ts, pupil);
	return;
}
