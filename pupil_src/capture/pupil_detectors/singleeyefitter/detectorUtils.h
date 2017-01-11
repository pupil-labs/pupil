#ifndef __DETECTORUTILS_H__
#define __DETECTORUTILS_H__


#include <opencv2/core.hpp>
#include "common/types.h"
#include "EllipseEvaluation2D.h"


namespace singleeyefitter {

    namespace detector {



        // returns lowest and highest intensity index and max intensity. only intensity values wich occure more than amount_intensity_values are taken in to account
        void calculate_spike_indices_and_max_intenesity(cv::Mat& histogram, int amount_intensity_values, int& lowest_spike_index, int& highest_spike_index, float& max_intensity);

        template< typename Scalar >
        Contours_2D split_rough_contours(const Contours_2D& contours, const Scalar max_angle);

        // splits contours in independent contours, satisfying that all single contours have the same curvature, at least min_contour_size points,
        // and the maximum angle between three points is max_angle
        template< typename Scalar >
        Contours_2D split_rough_contours_optimized(const Contours_2D& contours, const Scalar max_angle , const int min_contour_size);

        // returns the indices to strong and weak contours
        std::pair<ContourIndices, ContourIndices> divide_strong_and_weak_contours(
            const Contours_2D& contours, const EllipseEvaluation2D& is_ellipse, const float ellipse_fit_treshold,
            const float strong_perimeter_ratio_range_min, const float strong_perimeter_ratio_range_max,
            const float strong_area_ratio_range_min, const float strong_area_ratio_range_max);


        //calculates how much ellipse is supported by the contour
        // return the ratio of area and circumference of the ellipse to the contour
        std::pair<double, double> ellipse_contour_support_ratio(const Ellipse& ellipse, const Contour_2D& contour);

        //calculates how much the contour deviates from the fitted ellipse
        double contour_ellipse_deviation_variance(Contour_2D& contour);

        namespace detail {

            template<typename Scalar>
            std::vector<int> find_kink_and_dir_change(const std::vector<Scalar>& curvature, const Scalar max_angle);

            Contours_2D split_at_corner_index(const Contour_2D& contour, const std::vector<int>& indices);


        } //namespace detail

    } // namespace detector
}// namespace singleeyefitter

#endif // __DETECTORUTILS_H__
