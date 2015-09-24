
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct Result{
  int test;
  double test2;
};

Result detect( cv::Mat& image, cv::Rect& usr_roi, cv::Mat& color_image , bool use_color_image ){

  image = cv::Mat(image, usr_roi);  // image with usr_roi

  // coarse pupil detection
    //not implemented yet


  const int w = image.size().width;
  const float coarse_pupil_width = w/2.0f;
  const float padding = coarse_pupil_width/4.0f;

  const cv::Rect pupil_roi = usr_roi; // gets changed when coarse detection is on

  image = cv::Mat(image, pupil_roi ); // after coarse detection it in the region of the pupil

  cv::Mat histogram;
  int histSize, bins;
  bins = histSize = 256; //from 0 to 255
  /// Set the ranges
  float range[] = { 0, 256 } ; //the upper boundary is exclusive
  const float* histRange = { range };
  cv::calcHist( &image, 1 , 0, cv::Mat(), histogram , 1 , &histSize, &histRange, true, false );


  return { 1, 3.0 };
}
