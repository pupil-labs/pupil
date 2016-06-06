#ifndef __SINGLEEYEFITTER_CVX_H__
#define __SINGLEEYEFITTER_CVX_H__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace singleeyefitter {

    const double SQRT_2 = std::sqrt(2.0);
    const double PI = CV_PI;

    namespace cvx {

        inline cv::Scalar rgb(double r, double g, double b, double a = 0)
        {
            return cv::Scalar(b, g, r, a);
        }

        template<typename T>
        inline cv::Rect_<T> roiAround(T x, T y, T radius)
        {
            return cv::Rect_<T>(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1);
        }
        template<typename T>
        inline cv::Rect_<T> roiAround(const cv::Point_<T>& center, T radius)
        {
            return roiAround(center.x, center.y, radius);
        }


        inline cv::Mat& line(cv::Mat& dst, cv::Point2f from, cv::Point2f to, cv::Scalar color, int thickness = 1, int linetype = CV_AA, int shift = 8)
        {
            auto from_i = cv::Point(from.x * (1 << shift), from.y * (1 << shift));
            auto to_i = cv::Point(to.x * (1 << shift), to.y * (1 << shift));
            cv::line(dst, from_i, to_i, color, thickness, linetype, shift);
            return dst;
        }

        inline void cross(cv::Mat& img, cv::Point2f center, double radius, const cv::Scalar& colour, int thickness = 1, int lineType = CV_AA, int shift = 8)
        {
            cvx::line(img, center + cv::Point2f(-radius, -radius), center + cv::Point2f(radius, radius), colour, thickness, lineType, shift);
            cvx::line(img, center + cv::Point2f(-radius, radius), center + cv::Point2f(radius, -radius), colour, thickness, lineType, shift);
        }
        inline void plus(cv::Mat& img, cv::Point2f center, double radius, const cv::Scalar& colour, int thickness = 1, int lineType = CV_AA, int shift = 8)
        {
            cvx::line(img, center + cv::Point2f(0, -radius), center + cv::Point2f(0, radius), colour, thickness, lineType, shift);
            cvx::line(img, center + cv::Point2f(-radius, 0), center + cv::Point2f(radius, 0), colour, thickness, lineType, shift);
        }

        // find zero crashes if it doesn't find one. remove if opencv version is 3.0 or above
        inline void findNonZero( cv::Mat& in, cv::OutputArray& out ){
            // add a non zero value to the first pixel
            // This is a hack to avoid crash
            in.at<uchar>(0,0) = 1;
            cv::findNonZero(in, out);
            in.at<uchar>(0,0) = 0;
        }

        void draw_dotted_rect(cv::Mat& image, const cv::Rect& rect , const cv::Scalar& color);


        /*inline void cross(cv::Mat& img, cv::Point center, int radius, const cv::Scalar& colour, int thickness = 1, int lineType = 8, int shift = 0)
        {
            cv::line(img, center + cv::Point(-radius, -radius), center + cv::Point(radius, radius), colour, thickness, lineType, shift);
            cv::line(img, center + cv::Point(-radius, radius), center + cv::Point(radius, -radius), colour, thickness, lineType, shift);
        }
        inline void plus(cv::Mat& img, cv::Point center, int radius, const cv::Scalar& colour, int thickness = 1, int lineType = 8, int shift = 0)
        {
            cv::line(img, center + cv::Point(0, -radius), center + cv::Point(0, radius), colour, thickness, lineType, shift);
            cv::line(img, center + cv::Point(-radius, 0), center + cv::Point(radius, 0), colour, thickness, lineType, shift);
        }*/

        inline cv::Rect boundingBox(const cv::Mat& img)
        {
            return cv::Rect(0, 0, img.cols, img.rows);
        }

        // get the ROI where the boarder of black pixels is removed
        bool getRoiWithoutBorder(const cv::Mat& img, cv::Rect& roi);

        void getROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& roi, int borderType = cv::BORDER_REPLICATE);

        float histKmeans(const cv::Mat_<float>& hist, int bin_min, int bin_max, int K, float init_centers[], cv::Mat_<uchar>& labels, cv::TermCriteria termCriteria);

        cv::RotatedRect fitEllipse(const cv::Moments& m);
        cv::Vec2f majorAxis(const cv::RotatedRect& ellipse);


        inline cv::Mat resize(const cv::Mat& src, cv::Size size, int interpolation = cv::INTER_LINEAR)
        {
            cv::Mat dst;
            cv::resize(src, dst, size, 0, 0, interpolation);
            return dst;
        }
        inline cv::Mat resize(const cv::Mat& src, double fx, double fy = 0, int interpolation = cv::INTER_LINEAR)
        {
            if (fy == 0) fy = fx;

            cv::Mat dst;
            cv::resize(src, dst, cv::Size(), fx, fy, interpolation);
            return dst;
        }

        inline cv::Mat GaussianBlur(const cv::Mat& src, cv::Size ksize, double sigmax, double sigmay = 0, int borderType = cv::BORDER_DEFAULT)
        {
            cv::Mat dst;
            cv::GaussianBlur(src, dst, ksize, sigmax, sigmay, borderType);
            return dst;
        }
        inline cv::Mat GaussianBlur(cv::Mat&& src, cv::Size ksize, double sigmax, double sigmay = 0, int borderType = cv::BORDER_DEFAULT)
        {
            cv::GaussianBlur(src, src, ksize, sigmax, sigmay, borderType);
            return src;
        }
        inline cv::Mat GaussianBlur(const cv::Mat& src, double sigmax, double sigmay = 0, int borderType = cv::BORDER_DEFAULT)
        {
            return cvx::GaussianBlur(src, cv::Size(), sigmax, sigmay, borderType);
        }
        inline cv::Mat GaussianBlur(cv::Mat&& src, double sigmax, double sigmay = 0, int borderType = cv::BORDER_DEFAULT)
        {
            return cvx::GaussianBlur(src, cv::Size(), sigmax, sigmay, borderType);
        }

        inline cv::Mat convert(const cv::Mat& src, int rtype, double alpha = 1, double beta = 0)
        {
            cv::Mat dst;
            src.convertTo(dst, rtype, alpha, beta);
            return dst;
        }

        inline cv::Mat cvtColor(const cv::Mat& src, int code, int dstCn = 0)
        {
            cv::Mat dst;
            cv::cvtColor(src, dst, code, dstCn);
            return dst;
        }

        inline cv::Mat extractChannel(const cv::Mat& src, int coi)
        {
            cv::Mat dst;
            cv::extractChannel(src, dst, coi);
            return dst;
        }
    }

} //namespace singleeyefitter

#endif // __SINGLEEYEFITTER_CVX_H__
