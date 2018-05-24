#include "PupilDetectionMethod.h"
//#include <QDebug>

// TODO: clean up this interface and the one from the tracking
// this includes renaming everything from the old "run" method

using namespace std;
using namespace cv;

//#define DBG_COARSE_PUPIL_DETECTION
//#define DBG_OUTLINE_CONTRAST
//#include <QElapsedTimer>
Rect PupilDetectionMethod::coarsePupilDetection(const Mat &frame, const float &minCoverage, const int &workingWidth, const int &workingHeight)
{
	// We can afford to work on a very small input for haar features, but retain the aspect ratio
	float xr = frame.cols / (float) workingWidth;
	float yr = frame.rows / (float) workingHeight;
	float r = max( xr, yr );

	Mat downscaled;
    resize(frame, downscaled, Size(), 1/r, 1/r, CV_INTER_LINEAR);

	int ystep = (int) max<float>( 0.01f*downscaled.rows, 1.0f);
	int xstep = (int) max<float>( 0.01f*downscaled.cols, 1.0f);

	float d = (float) sqrt( pow(downscaled.rows, 2) + pow(downscaled.cols, 2) );

	// Pupil radii is based on PuRe assumptions
	int min_r = (int) (0.5 * 0.07 * d);
	int max_r = (int) (0.5 * 0.29 * d);
	int r_step = (int) max<float>( 0.2f*(max_r + min_r), 1.0f);

	// TODO: padding so we consider the borders as well!

	/* Haar-like feature suggested by Swirski. For details, see
	 * Świrski, Lech, Andreas Bulling, and Neil Dodgson.
	 * "Robust real-time pupil tracking in highly off-axis images."
	 * Proceedings of the Symposium on Eye Tracking Research and Applications. ACM, 2012.
	 *
	 * However, we collect a per-pixel maxima instead of the global one
	*/
	Mat itg;
	integral(downscaled, itg, CV_32S);
    Mat res = Mat::zeros( downscaled.rows, downscaled.cols, CV_32F);
	float best_response = std::numeric_limits<float>::min();
    deque< pair<Rect, float> > candidates;
	for (int r = min_r; r<=max_r; r+=r_step) {
		int step = 3*r;

		Point ia, ib, ic, id;
		Point oa, ob, oc, od;

		int inner_count = (2*r) * (2*r);
		int outer_count = (2*step)*(2*step) - inner_count;

		float inner_norm = 1.0f / (255*inner_count);
		float outer_norm = 1.0f / (255*outer_count);

		for (int y = step; y<downscaled.rows-step; y+=ystep) {
			oa.y = y - step;
			ob.y = y - step;
			oc.y = y + step;
			od.y = y + step;
			ia.y = y - r;
			ib.y = y - r;
			ic.y = y + r;
			id.y = y + r;
            for (int x = step; x<downscaled.cols-step; x+=xstep) {
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

				float inner_mean = inner_norm*inner;
				float outer_mean = outer_norm*outer;
                float response = (outer_mean - inner_mean);

                if ( response < 0.5*best_response)
                    continue;

				if (response < 0.5 * best_response)
					continue;

				if (response > best_response)
					best_response = response;

				if ( response > res.ptr<float>(y)[x] ) {
					res.ptr<float>(y)[x] = response;
					// The pupil is too small, the padding too large; we combine them.
					candidates.push_back( make_pair(Rect( 0.5*(ia+oa), 0.5*(ic+oc) ), response) );
				}

			}
		}
	}

	auto compare = [] (const pair<Rect, float> &a, const pair<Rect,float> &b) {
		return (a.second > b.second);
	};
	sort( candidates.begin(), candidates.end(), compare);

#ifdef DBG_COARSE_PUPIL_DETECTION
	Mat dbg;
	cvtColor(downscaled, dbg, CV_GRAY2BGR);
#endif

	// Now add until we reach the minimum coverage or run out of candidates
	Rect coarse;
	int minWidth = minCoverage * downscaled.cols;
	int minHeight = minCoverage * downscaled.rows;
	for ( int i=0; i<candidates.size(); i++ ) {
		auto &c = candidates[i];
		if (coarse.area() == 0)
			coarse = c.first;
		else
			coarse |= c.first;
#ifdef DBG_COARSE_PUPIL_DETECTION
		rectangle(dbg, candidates[i].first, Scalar(0,255,255));
#endif
		if (coarse.width > minWidth && coarse.height > minHeight)
			break;
    }

#ifdef DBG_COARSE_PUPIL_DETECTION
	rectangle(dbg, coarse, Scalar(0,255,0));
	resize(dbg, dbg, Size(), r, r);
	imshow("Coarse Detection Debug", dbg);
#endif

	// Upscale result
	coarse.x *= r;
	coarse.y *= r;
	coarse.width *= r;
	coarse.height *= r;

	// Sanity test
	Rect imRoi = Rect(0, 0, frame.cols, frame.rows);
	coarse &= imRoi;
	if (coarse.area() == 0)
		return imRoi;

	return coarse;
}

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

static void inline sincos(int angle, float& cosval, float& sinval)
{
	angle += (angle < 0 ? 360 : 0);
	sinval = sinTable[angle];
	cosval = sinTable[450 - angle];
}

static inline vector<Point> ellipse2Points(const RotatedRect &ellipse, const int &delta=1)
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

/* Measures the confidence for a pupil based on the inner-outer contrast
 * from the pupil following PuRe. For details, see
 * Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci
 * "PuRe: Robust pupil detection for real-time pervasive eye tracking"
 * Under review on Elsevier's Computer Vision and Image Understanding journal.
 * TODO: update when published
 */
float PupilDetectionMethod::outlineContrastConfidence(const Mat &frame, const Pupil &pupil, const int &bias)
{
	if ( ! pupil.hasOutline() )
		return NO_CONFIDENCE;

	Rect boundaries = { 0, 0, frame.cols, frame.rows };
	int minorAxis = min<int>(pupil.size.width, pupil.size.height);
	int delta = 0.15*minorAxis;
	cv::Point c = pupil.center;

#ifdef DBG_OUTLINE_CONTRAST
	cv::Mat tmp;
	cv::cvtColor(frame, tmp, CV_GRAY2BGR);
	cv::ellipse(tmp, pupil, cv::Scalar(0,255,255));
	cv::Scalar lineColor;
#endif
	int evaluated = 0;
	int validCount = 0;

	vector<Point> outlinePoints = ellipse2Points(pupil, 10);
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
				m1 += frame.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int x=p->x+1; x<=ex; x++) {
				m2 += frame.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
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
				m1 += frame.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int y=p->y+1; y<=ey; y++)
				m2 += frame.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
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
		return 0;

#ifdef DBG_OUTLINE_CONTRAST
	cv::imshow("Outline Contrast Debug", tmp);
#endif

	return validCount / (float) evaluated;
}

float PupilDetectionMethod::angularSpreadConfidence(const vector<Point> &points, const Point2f &center)
{
	enum {
		Q0 = 0,
		Q1 = 1,
		Q2 = 2,
		Q3 = 3,
	};

	std::bitset<4> anchorPointSlices;
	anchorPointSlices.reset();
	for (auto p=points.begin(); p!=points.end(); p++) {
		if (p->x - center.x < 0) {
			if (p->y - center.y < 0)
				anchorPointSlices.set(Q0);
			else
				anchorPointSlices.set(Q3);
		} else  {
			if (p->y - center.y < 0)
				anchorPointSlices.set(Q1);
			else
				anchorPointSlices.set(Q2);
		}
	}
	return anchorPointSlices.count() / (float) anchorPointSlices.size();
}

float PupilDetectionMethod::aspectRatioConfidence(const Pupil &pupil)
{
	return pupil.minorAxis() / (float) pupil.majorAxis();
}

float PupilDetectionMethod::edgeRatioConfidence(const Mat &edgeImage, const Pupil &pupil, vector<Point> &edgePoints, const int &band)
{
	if (!pupil.valid())
		return NO_CONFIDENCE;
	Mat outlineMask = Mat::zeros(edgeImage.rows, edgeImage.cols, CV_8U);
	ellipse(outlineMask, pupil, Scalar(255), band);
	Mat inBandEdges = edgeImage.clone();
	inBandEdges.setTo(0, 255 - outlineMask);
	findNonZero(inBandEdges, edgePoints);
	return min<float>( edgePoints.size() / pupil.circumference(), 1.0 );
}

