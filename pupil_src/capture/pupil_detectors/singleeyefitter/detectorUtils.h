#ifndef __DETECTORUTILS_H__
#define __DETECTORUTILS_H__


#include <opencv2/core/core.hpp>
#include "common/types.h"

namespace singleeyefitter {

namespace detector {



  // returns lowest and highest intensity index and max intensity. only intensity values wich occure more than amount_intensity_values are taken in to account
  void calculate_spike_indices_and_max_intenesity( cv::Mat& histogram, int amount_intensity_values, int& lowest_spike_index, int& highest_spike_index, float& max_intensity );

  template< typename Scalar >
  Contours_2D split_contours( const Contours_2D& contours, const Scalar angle );
  template< typename Scalar >
  Contours_2D split_contours_optimized( const Contours_2D& contours, const Scalar angle , const int min_contour_size);

  namespace detail{


    template<typename Scalar>
    std::vector<int> find_kink_and_dir_change(const std::vector<Scalar>& curvature, const Scalar max_angle);

    Contours_2D split_at_corner_index(const Contour_2D& contour,const std::vector<int>& indices);


  } //namespace detail

} // namespace detector
}// namespace singleeyefitter

#endif // __DETECTORUTILS_H__
