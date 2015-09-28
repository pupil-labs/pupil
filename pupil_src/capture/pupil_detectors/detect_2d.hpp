
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "singleeyefitter/utils.h"
#include "singleeyefitter/cvx.h"
#include "singleeyefitter/Ellipse.h"  // use ellipse eyefitter
#include "singleeyefitter/distance.h"
#include <iostream>


typedef singleeyefitter::Ellipse2D<double> Ellipse;

struct Result{
  double confidence =  0.0 ;
  Ellipse ellipse;
  double timeStampe = 0.0;
};

// use a struct for all properties and pass it to detect method every time we call it.
// Thus we don't need to keep track if GUI is updated and cython handles conversion from Dict to struct
struct DetectProperties{
  int intensity_range;
  int blur_size;
  float canny_treshold;
  float canny_ration;
  int canny_aperture;
  int pupil_size_max;
  int pupil_size_min;
  float strong_perimeter_ratio_range_min;
  float strong_perimeter_ratio_range_max;
  int contour_size_min;
};

using namespace singleeyefitter;

class Detector2D{

public:

  Detector2D();
  Result detect(DetectProperties& props, cv::Mat& image, cv::Mat& color_image, cv::Mat& debug_image, cv::Rect& usr_roi , bool visualize, bool use_debug_image);
  std::vector<cv::Point> ellipse_true_support(Ellipse& ellipse, double ellipse_circumference, std::vector<cv::Point>& raw_edges);


private:

  bool mUse_strong_prior;
  int mPupil_Size;
  Ellipse mPrior_ellipse;

  const cv::Scalar mRed_color = {0,0,255};
  const cv::Scalar mGreen_color = {0,255,0};
  const cv::Scalar mBlue_color = {255,0,0};
  const cv::Scalar mRoyalBlue_color = {255,100,100};
  const cv::Scalar mYellow_color = {255,255,0};
  const cv::Scalar mWhite_color = {255,255,255};

};

Detector2D::Detector2D(): mUse_strong_prior(false), mPupil_Size(100){};


std::vector<cv::Point> Detector2D::ellipse_true_support( Ellipse& ellipse, double ellipse_circumference, std::vector<cv::Point>& raw_edges){

  double major_radius = ellipse.major_radius;
  double minor_radius = ellipse.minor_radius;

  std::cout << ellipse_circumference << std::endl;
  std::vector<double> distances;
  std::vector<cv::Point> support_pixels;

  for(auto it = raw_edges.begin(); it != raw_edges.end(); it++){

      const cv::Point p = *it;
      double distance = euclidean_distance( (double)p.x, (double)p.y, ellipse);

      if(distance <=  1.3 ){
        support_pixels.push_back(p);
      }
  }

  return std::move(support_pixels);
}

Result Detector2D::detect( DetectProperties& props, cv::Mat& image, cv::Mat& color_image, cv::Mat& debug_image, cv::Rect& usr_roi , bool visualize, bool use_debug_image ){

  Result result;

  image = cv::Mat(image, usr_roi);  // image with usr_roi

  const int image_width = image.size().width;
  const int image_height = image.size().height;

  const int w = image.size().width/2;
  const float coarse_pupil_width = w/2.0f;
  const int padding = int(coarse_pupil_width/4.0f);

  const int offset = props.intensity_range;
  const int spectral_offset = 5;

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

      const int scale_x  = 100;
      const int scale_y = 1 ;

      for(int i = 0; i < histogram.rows; i++){
        const float norm_i  = histogram.ptr<float>(i)[0]/max_intensity ; // normalized intensity
        cv::line( color_image, {image_width, i*scale_y}, { image_width - int(norm_i * scale_x), i * scale_y}, mBlue_color );
      }
      cv::line(color_image, {image_width, lowest_spike_index* scale_y}, { int(image_width - 0.5f * scale_x), lowest_spike_index * scale_y }, mRed_color);
      cv::line(color_image, {image_width, (lowest_spike_index+offset)* scale_y}, { int(image_width - 0.5f * scale_x), (lowest_spike_index + offset)* scale_y }, mYellow_color);
      cv::line(color_image, {image_width, (highest_spike_index)* scale_y}, { int(image_width - 0.5f * scale_x), highest_spike_index* scale_y }, mRed_color);
      cv::line(color_image, {image_width, (highest_spike_index - spectral_offset)* scale_y}, { int(image_width - 0.5f * scale_x), (highest_spike_index - spectral_offset)* scale_y }, mWhite_color);

  }

   //create dark and spectral glint masks
  cv::Mat binary_img;
  cv::Mat kernel;
  cv::inRange( image, cv::Scalar(0) , cv::Scalar(lowest_spike_index + props.intensity_range), binary_img );  // binary threshold
  kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7,7} );
  cv::dilate( binary_img, binary_img, kernel, {-1,-1}, 2 );

  cv::Mat spec_mask;
  cv::inRange( image, cv::Scalar(0) , cv::Scalar(highest_spike_index + spectral_offset), spec_mask );  // binary threshold
  cv::erode( spec_mask, spec_mask, kernel);

  kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9,9} );
  //open operation to remove eye lashes
  cv::morphologyEx( image, image, cv::MORPH_OPEN, kernel);

  if( props.blur_size > 1 )
    cv::medianBlur(image, image, props.blur_size );

  cv::Mat edges;
  cv::Canny( image, edges, props.canny_treshold, props.canny_treshold * props.canny_ration, props.canny_aperture);

  //remove edges in areas not dark enough and where the glint is (spectral refelction from IR leds)
  cv::min(edges, spec_mask, edges);
  cv::min(edges, binary_img, edges);


  if( visualize ){
    // get sub matrix
    cv::Mat overlay = color_image.colRange(usr_roi.x + pupil_roi.x, usr_roi.x + pupil_roi.x + pupil_roi.width).rowRange(usr_roi.y + pupil_roi.y, usr_roi.y + pupil_roi.y + pupil_roi.height);
    cv::Mat g_channel( overlay.rows, overlay.cols, CV_8UC1 );
    cv::Mat b_channel( overlay.rows, overlay.cols, CV_8UC1 );
    cv::Mat r_channel( overlay.rows, overlay.cols, CV_8UC1 );
    cv::Mat out[] = {b_channel, g_channel,r_channel};
    cv:split(overlay, out);

    cv::max(g_channel, edges,g_channel);
    cv::max(b_channel, binary_img,b_channel);
    cv::min(b_channel, spec_mask,b_channel);

    cv::merge(out, 3, overlay);

    //draw a frame around the automatic pupil ROI in overlay.
    auto rect = cv::Rect(pupil_roi.x+2, pupil_roi.y+2, pupil_roi.width-2, pupil_roi.height-2);
    cvx::draw_dotted_rect( overlay, rect, 255);
    //draw a frame around the area we require the pupil center to be.
    rect = cv::Rect(padding, padding, pupil_roi.width-padding, pupil_roi.height-padding);
    cvx::draw_dotted_rect( overlay, rect, 255);

    //draw size ellipses
    cv::Point center(100, image_height -100);
    cv::circle( overlay, center, props.pupil_size_min/2.0, mRed_color );
    cv::circle( overlay, center, mPupil_Size/2.0, mGreen_color );           // real pupil size of this frame is calculated further down, so this size is from the last frame
    cv::circle( overlay, center, props.pupil_size_max/2.0, mRed_color );

    auto text_string = std::to_string(mPupil_Size);
    cv::Size text_size = cv::getTextSize( text_string, cv::FONT_HERSHEY_SIMPLEX, 0.4 , 1, 0);
    cv::Point text_pos = { center.x - text_size.width/2 , center.y + text_size.height/2};
    cv::putText( overlay, text_string, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.4, mRoyalBlue_color );


  }

  //get raw edge pix for later
  std::vector<cv::Point> raw_edges;
  cv::findNonZero(edges, raw_edges);

  ///////////////////////////////
  /// Strong Prior Part Begin ///
  ///////////////////////////////

  //if we had a good ellipse before ,let see if it is still a good first guess:
  if( mUse_strong_prior ){

    mUse_strong_prior = false;

    //recalculate center in coords system of ROI views
    Ellipse ellipse = mPrior_ellipse;
    ellipse.center[0] -= ( pupil_roi.x + usr_roi.x  );
    ellipse.center[1] -= ( pupil_roi.y + usr_roi.y  );

    if( !raw_edges.empty() ){

      std::vector<cv::Point> support_pixels;
      double ellipse_circumference = ellipse.circumference();
      support_pixels = ellipse_true_support(ellipse, ellipse_circumference, raw_edges);

      double support_ration = support_pixels.size() / ellipse_circumference;

      if(support_ration >= props.strong_perimeter_ratio_range_min){

        cv::RotatedRect refit_ellipse = cv::fitEllipse(support_pixels);

        if(use_debug_image){
            cv::ellipse(debug_image, toRotatedRect(ellipse), mRoyalBlue_color, 4);
            cv::ellipse(debug_image, refit_ellipse, mRed_color, 1);
        }

        ellipse = toEllipse<double>(refit_ellipse);
        ellipse.center[0] += ( pupil_roi.x + usr_roi.x  );
        ellipse.center[1] += ( pupil_roi.y + usr_roi.y  );
        mPrior_ellipse = ellipse;

        double goodness = std::min(0.1, support_ration);

        result.confidence = goodness;
        result.ellipse = ellipse;

        mPupil_Size = ellipse.major_radius;

        return result;

      }
    }
  }

  ///////////////////////////////
  ///  Strong Prior Part End  ///
  ///////////////////////////////

  //from edges to contours
  std::vector<std::vector<cv::Point>> contours, good_contours;
  cv::findContours(edges, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
  good_contours.resize( contours.size() );

  //first we want to filter out the bad stuff, to short
  auto min_contour_size_pred = [&]( const std::vector<cv::Point>& contour){ return contour.size() > props.contour_size_min; };
  std::copy_if( contours.begin(), contours.end(), good_contours.begin(), min_contour_size_pred); // better way than copy, erase probably not better with vector

  //now we learn things about each contour through looking at the curvature.
  //For this we need to simplyfy the contour so that pt to pt angles become more meaningfull

  cv::imshow("asdasd", edges );
  cv::imshow("debug_image", debug_image);

  return result;

}
