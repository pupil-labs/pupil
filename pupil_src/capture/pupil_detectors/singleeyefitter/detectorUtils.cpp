
#include "detectorUtils.h"
#include "mathHelper.h"

#include <vector>

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

template< typename Scalar >
Contours_2D detector::split_contours( const Contours_2D& contours, const Scalar angle ){

  Contours_2D split_contours;

  for(auto it = contours.begin(); it != contours.end(); it++ ){

    const Contour_2D& contour  = *it;
    // keeps track of angles between 3 adjacent points
    std::vector<Scalar> curvature;
    // closed curves not handled yet
    for(auto point_it = contour.begin(); point_it != contour.end()-2; point_it++){
        const cv::Point& first = *point_it;
        const cv::Point& second = *(point_it+1);
        const cv::Point& third = *(point_it+2);
        curvature.push_back( math::getAngleABC<float>(first, second, third) );
    }

    //we split whenever there is a real kink (abs(curvature)<right angle) or a change in the genreal direction

    auto kink_indices = detail::find_kink_and_dir_change( curvature, angle);
    auto contour_segments = detail::split_at_corner_index( contour, kink_indices);
    //TODO: split at shart inward turns
    int colorIndex = 0;
    for( auto seg_it = contour_segments.begin(); seg_it != contour_segments.end(); seg_it++){

      std::vector<cv::Point> segment = *seg_it;
      //printPoints(segment);
      if( segment.size() > 2 ){
        split_contours.push_back(segment);

        // // debug segments
        // if(use_debug_image){
        //   const cv::Scalar_<int> colors[] = {mRed_color, mBlue_color, mRoyalBlue_color, mYellow_color, mWhite_color, mGreen_color};
        //   cv::polylines(debug_image, segment, false, colors[colorIndex], 1, 4);
        //   colorIndex++;
        //   colorIndex %= 6;
        // }
      }
    }
  }

  return split_contours;

}

template<typename Scalar>
std::vector<int> detector::detail::find_kink_and_dir_change(const std::vector<Scalar>& curvature, const Scalar max_angle){

  std::vector<int> split_indeces;
  if( curvature.empty() ) return split_indeces;

  bool currently_positive = curvature.at(0) > 0;
  for(int i=0 ; i < curvature.size(); i++){
      Scalar angle = curvature.at(i);
      bool is_positive = angle > 0;
      if( std::abs(angle) < max_angle || is_positive != currently_positive ){
        currently_positive = is_positive;
        split_indeces.push_back(i);
      }
  }
  return split_indeces;
}
Contours_2D detector::detail::split_at_corner_index(const Contour_2D& contour,const std::vector<int>& indices){

  std::vector<std::vector<cv::Point>> contour_segments;
  if(indices.empty()){
     contour_segments.push_back(contour);
      return contour_segments;
  }
  int startIndex = 0;
  for(int i=0 ; i < indices.size() + 1 ; i++){
      int next_Index = i < indices.size() ?  indices.at(i) + 1  : contour.size()-1; // don't forget the last one
      auto begin = contour.begin();
      contour_segments.push_back( {begin + startIndex , begin + next_Index + 1} );
      startIndex = next_Index;
  }
  return contour_segments;
}



// tell the compile to generate these to explicit templates, otherwise it wouldn't know which one to create at comile time
template Contours_2D detector::split_contours( const Contours_2D& contours, const float angle );
template Contours_2D detector::split_contours( const Contours_2D& contours, const double angle );



} // singleeyefitter namespace
