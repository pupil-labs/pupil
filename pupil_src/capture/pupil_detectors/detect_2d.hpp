
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

struct Result{
  int test;
  double test2;
};

Result detect( cv::Mat& image, cv::Rect& usr_roi, cv::Mat& color_image , bool visualize , int intensity_range ){

  image = cv::Mat(image, usr_roi);  // image with usr_roi

  // coarse pupil detection
    //not implemented yet


  const int w = image.size().width;
  const float coarse_pupil_width = w/2.0f;
  const float padding = coarse_pupil_width/4.0f;

  const cv::Rect pupil_roi = usr_roi; // gets changed when coarse detection is on

  image = cv::Mat(image, pupil_roi ); // after coarse detection it in the region of the pupil

  cv::Mat histogram;
  int histSize;
  histSize = 256; //from 0 to 255
  /// Set the ranges
  float range[] = { 0, 256 } ; //the upper boundary is exclusive
  const float* histRange = { range };
  cv::calcHist( &image, 1 , 0, cv::Mat(), histogram , 1 , &histSize, &histRange, true, false );

  std::vector<int> spikes_index;
  spikes_index.reserve(histogram.rows);
  int lowest_spike_index = 255;
  int highest_spike_index = 0;
  float max_intensity = 0;
  //  every intensity seen in more than 40 pixels
  for(int i = 0; i < histogram.rows; i++ )
  {
      const float intensity  = histogram.at<float>(i,0);

      if( intensity > 40 ){
        max_intensity = std::max(intensity, max_intensity);
        spikes_index.push_back(i);
        lowest_spike_index = std::min(lowest_spike_index, i);
        highest_spike_index = std::max( highest_spike_index, i);
      }
  }

  if(spikes_index.size() == 0 ){
     lowest_spike_index = 200;
     highest_spike_index = 255;
  }

  if( visualize ){  // if visualize is true, we draw to the color image

      const int width = color_image.size().width;
      const int height = color_image.size().height;
      const int scale_x  = 100;
      const int scale_y = 1 ;
      const int offset = intensity_range;
      const int spectral_offset = 5;

      const cv::Scalar colors[4] = {{0,0,255},{255,0,0},{255,255,0},{255,255,255}};
      for(int i = 0; i < histogram.rows; i++){
        const float norm_i  = histogram.ptr<float>(i)[0]/max_intensity ; // normalized intensity
        cv::line( color_image, {width, i*scale_y}, { width - int(norm_i * scale_x), i * scale_y}, colors[1] );
      }
      cv::line(color_image, {width, lowest_spike_index* scale_y}, { int(width - 0.5f * scale_x), lowest_spike_index * scale_y },colors[0]);
      cv::line(color_image, {width, (lowest_spike_index+offset)* scale_y}, { int(width - 0.5f * scale_x), (lowest_spike_index + offset)* scale_y },colors[2]);
      cv::line(color_image, {width, (highest_spike_index)* scale_y}, { int(width - 0.5f * scale_x), highest_spike_index* scale_y },colors[0]);
      cv::line(color_image, {width, (highest_spike_index - spectral_offset)* scale_y}, { int(width - 0.5f * scale_x), (highest_spike_index - spectral_offset)* scale_y },colors[3]);

  }



  return { 1, 3.0 };

}
