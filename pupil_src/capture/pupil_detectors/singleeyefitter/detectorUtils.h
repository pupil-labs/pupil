#ifndef __DETECTORUTILS_H__
#define __DETECTORUTILS_H__


#include <opencv2/core/core.hpp>

namespace singleeyefitter {

namespace detector {



  // returns lowest and highest intensity index and max intensity. only intensity values wich occure more than amount_intensity_values are taken in to account
  void calculate_spike_indices_and_max_intenesity( cv::Mat& histogram, int amount_intensity_values, int& lowest_spike_index, int& highest_spike_index, float& max_intensity );

} // namespace detector
}// namespace singleeyefitter

#endif // __DETECTORUTILS_H__
