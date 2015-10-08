#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <stdexcept>

#include <iostream>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "Ellipse.h"


namespace singleeyefitter {

	class MakeString {
		public:
			std::stringstream stream;
			operator std::string() const { return stream.str(); }

			template<class T>
			MakeString& operator<<(T const& VAR) { stream << VAR; return *this; }
	};

	inline int pow2(int n)
	{
		return 1 << n;
	}

	int random(int min, int max);
	int random(int min, int max, unsigned int seed);

	template<typename T>
	std::vector<T> randomSubset(const std::vector<T>& src, typename std::vector<T>::size_type size)
	{
		if (size > src.size())
			throw std::range_error("Subset size out of range");

		std::vector<T> ret;
		std::set<size_t> vals;

		for (size_t j = src.size() - size; j < src.size(); ++j) {
			size_t idx = random(0, j); // generate a random integer in range [0, j]

			if (vals.find(idx) != vals.end())
				idx = j;

			ret.push_back(src[idx]);
			vals.insert(idx);
		}

		return ret;
	}

	template<typename T>
	std::vector<T> randomSubset(const std::vector<T>& src, typename std::vector<T>::size_type size, unsigned int seed)
	{
		if (size > src.size())
			throw std::range_error("Subset size out of range");

		std::vector<T> ret;
		std::set<size_t> vals;

		for (size_t j = src.size() - size; j < src.size(); ++j) {
			size_t idx = random(0, j, seed + j); // generate a random integer in range [0, j]

			if (vals.find(idx) != vals.end())
				idx = j;

			ret.push_back(src[idx]);
			vals.insert(idx);
		}

		return ret;
	}

	template<typename Scalar>
	inline Eigen::Matrix<Scalar, 2, 1> toEigen(const cv::Point2f& point)
	{
		return Eigen::Matrix<Scalar, 2, 1>(static_cast<Scalar>(point.x),
		                                   static_cast<Scalar>(point.y));
	}
	template<typename Scalar>
	inline cv::Point2f toPoint2f(const Eigen::Matrix<Scalar, 2, 1>& point)
	{
		return cv::Point2f(static_cast<float>(point[0]),
		                   static_cast<float>(point[1]));
	}
	template<typename Scalar>
	inline cv::Point toPoint(const Eigen::Matrix<Scalar, 2, 1>& point)
	{
		return cv::Point(static_cast<int>(point[0]),
		                 static_cast<int>(point[1]));
	}
	template<typename Scalar>
	inline cv::RotatedRect toRotatedRect(const Ellipse2D<Scalar>& ellipse)
	{
		return cv::RotatedRect(toPoint2f(ellipse.center),
		                       cv::Size2f(static_cast<float>(2.0 * ellipse.major_radius),
		                                  static_cast<float>(2.0 * ellipse.minor_radius)),
		                       static_cast<float>(ellipse.angle * 180.0 / M_PI));
	}
	template<typename Scalar>
	inline Ellipse2D<Scalar> toEllipse(const cv::RotatedRect& rect)
	{
		// Scalar major = rect.size.height;
		// Scalar minor = rect.size.width;
		// if(major < minor ){
		//     std::cout << "Flip major minor !!" << std::endl;
		//     std::swap(major,minor);
		// }
		return Ellipse2D<Scalar>(toEigen<Scalar>(rect.center),
		                         static_cast<Scalar>(rect.size.height / 2.0),
		                         static_cast<Scalar>(rect.size.width / 2.0),
		                         static_cast<Scalar>((rect.angle + 90.0) * M_PI / 180.0));
	}

} //namespace singleeyefitter

#endif // __UTILS_H__
