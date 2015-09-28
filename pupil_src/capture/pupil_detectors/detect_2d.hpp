
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "singleeyefitter/Ellipse.h"  // use ellipse eyefitter

#include <iostream>

struct Result{
  int test;
  double test2;
};

// use a struct for all properties and pass it to detect method every time we call it.
// Thus we don't need to keep track if GUI is updated and cython handles conversion from Dict to struct
struct DetectProperties{
  int intensity_range;
  int blur_size;
  double canny_treshold;
  double canny_ration;
  int canny_aperture;
  int pupil_max;
  int pupil_min;
  int target_size;
};

typedef singleeyefitter::Ellipse2D<double> Ellipse;

class Detector2D{

public:

  Detector2D();
  Result detect(cv::Mat& image, cv::Rect& usr_roi, cv::Mat& color_image , bool visualize, DetectProperties& props);



private:

  void draw_dotted_rect( cv::Mat& image, cv::Rect& rect , int color );

  bool mUse_strong_prior;
  Ellipse mPrior_ellipse;

  const cv::Scalar mRed_color = {0,0,255};
  const cv::Scalar mGreen_color = {0,255,0};
  const cv::Scalar mBlue_color = {255,0,0};
  const cv::Scalar mYellow_color = {255,255,0};
  const cv::Scalar mWhite_color = {255,255,255};

};

Detector2D::Detector2D(): mUse_strong_prior(false){};


void Detector2D::draw_dotted_rect( cv::Mat& image, cv::Rect& rect , int color ){

  int count = 0;
  auto create_Dotted_Line = [&](cv::Vec3b& pixel){
      if( count%4 == 0)
        pixel[0] = pixel[1] = pixel[2] = color;
      count++;
  };
  int x = rect.x;
  int y = rect.y;
  int width = rect.width-1;
  int height = rect.height-1;

  cv::Mat line  = image.colRange(x, width +1 ).rowRange( y , y +1);
  cv::Mat line2  = image.colRange(x, x +1 ).rowRange( y ,height +1);
  cv::Mat line3  = image.colRange(x, width +1 ).rowRange( height , height +1);
  cv::Mat line4  = image.colRange(width, width +1 ).rowRange( y , height+1);
  std::for_each(line.begin<cv::Vec3b>(), line.end<cv::Vec3b>(), create_Dotted_Line );
  count = 0;
  std::for_each(line2.begin<cv::Vec3b>(), line2.end<cv::Vec3b>(), create_Dotted_Line );
  count = 0;
  std::for_each(line3.begin<cv::Vec3b>(), line3.end<cv::Vec3b>(), create_Dotted_Line );
  count = 0;
  std::for_each(line4.begin<cv::Vec3b>(), line4.end<cv::Vec3b>(), create_Dotted_Line );

}

Result Detector2D::detect( cv::Mat& image, cv::Rect& usr_roi, cv::Mat& color_image , bool visualize, DetectProperties& props  ){

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
    draw_dotted_rect( overlay, rect, 255);
    //draw a frame around the area we require the pupil center to be.
    rect = cv::Rect(padding, padding, pupil_roi.width-padding, pupil_roi.height-padding);
    draw_dotted_rect( overlay, rect, 255);

    //draw size ellipses
    cv::Point center(100, image_height -100);
    cv::circle( overlay, center, props.pupil_min/2.0, mRed_color );
    cv::circle( overlay, center, props.target_size/2.0, mGreen_color );
    cv::circle( overlay, center, props.pupil_max/2.0, mRed_color );

    auto text_string = std::to_string(props.target_size);
    cv::Size text_size = cv::getTextSize( text_string, cv::FONT_HERSHEY_SIMPLEX, 0.4 , 1, 0);
    cv::Point text_pos = { center.x - text_size.width/2 , center.y + text_size.height/2};
    cv::putText( overlay, text_string, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.4, {255,100,100} );


  }

  //get raw edge pix for later
  cv::Mat raw_edges;
  cv::findNonZero(edges, raw_edges);

  cv::imshow("asdasd", edges );
  return { 1, 3.0 };

}
