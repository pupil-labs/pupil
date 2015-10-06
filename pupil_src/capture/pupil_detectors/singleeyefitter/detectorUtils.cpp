#include "detectorUtils.h"



namespace singleeyefitter{

 void detector::calculate_spike_indices_and_max_intenesity(
    cv::Mat& histogram, int amount_intensity_values, int& lowest_spike_index, int& highest_spike_index, float& max_intensity ){

    lowest_spike_index = 255;
    highest_spike_index = 0;
    max_intensity = 0;
    bool found_one = false;
    for(int i = 0; i < histogram.rows; i++ )
    {
        const float intensity  = histogram.at<float>(i,0);

        //  every intensity seen in more than amount_intensity_values pixels
        if( intensity > amount_intensity_values ){
          max_intensity = std::max(intensity, max_intensity);
          lowest_spike_index = std::min(lowest_spike_index, i);
          highest_spike_index = std::max( highest_spike_index, i);
          found_one = true;
        }
    }
    if( !found_one ){
       lowest_spike_index = 200;
       highest_spike_index = 255;
    }
  }


} // singleeyefitter namespace
