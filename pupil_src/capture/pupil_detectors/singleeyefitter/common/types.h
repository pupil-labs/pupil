
#ifndef singleeyefitter_types_h__
#define singleeyefitter_types_h__

#include "../Ellipse.h"

#include <vector>
#include <opencv2/core/core.hpp>


namespace singleeyefitter {

	typedef std::vector<std::vector<cv::Point>> Contours_2D;
	typedef std::vector<cv::Point> Contour_2D;
	typedef std::vector<int> ContourIndices;
	typedef singleeyefitter::Ellipse2D<double> Ellipse;

} // singleeyefitter namespace

#endif //singleeyefitter_types_h__
