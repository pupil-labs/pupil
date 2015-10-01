
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


#include "singleeyefitter/utils.h"
#include "singleeyefitter/cvx.h"
#include "singleeyefitter/Ellipse.h"  // use ellipse eyefitter
#include "singleeyefitter/distance.h"
#include "singleeyefitter/mathHelper.h"
#include "singleeyefitter/EllipseDistanceApproxCalculator.h"

template<typename Scalar>
struct Result{
  typedef singleeyefitter::Ellipse2D<Scalar> Ellipse;
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
  float strong_area_ratio_range_min;
  float strong_area_ratio_range_max;
  int contour_size_min;
  float ellipse_roundness_ratio;
  float initial_ellipse_fit_treshhold;

};


template< typename Scalar >
class Detector2D{

private:
  typedef singleeyefitter::Ellipse2D<Scalar> Ellipse;


public:

  Detector2D();
  Result<Scalar> detect(DetectProperties& props, cv::Mat& image, cv::Mat& color_image, cv::Mat& debug_image, cv::Rect& usr_roi , bool visualize, bool use_debug_image);
  std::vector<cv::Point> ellipse_true_support(Ellipse& ellipse, Scalar ellipse_circumference, std::vector<cv::Point>& raw_edges);


private:

  bool mUse_strong_prior;
  int mPupil_Size;
  Ellipse mPrior_ellipse;

  const cv::Scalar_<int> mRed_color = {0,0,255};
  const cv::Scalar_<int> mGreen_color = {0,255,0};
  const cv::Scalar_<int> mBlue_color = {255,0,0};
  const cv::Scalar_<int> mRoyalBlue_color = {255,100,100};
  const cv::Scalar_<int> mYellow_color = {255,255,0};
  const cv::Scalar_<int> mWhite_color = {255,255,255};

};

void printPoints( std::vector<cv::Point> points){
  std::for_each(points.begin(), points.end(), [](cv::Point& p ){ std::cout << p << std::endl;} );
}

template<typename Scalar>
std::vector<int> find_kink_and_dir_change(std::vector<Scalar>& curvature, float max_angle){

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
std::vector<std::vector<cv::Point>> split_at_corner_index(std::vector<cv::Point>& contour, std::vector<int>& indices){

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
template <typename Scalar>
Detector2D<Scalar>::Detector2D(): mUse_strong_prior(false), mPupil_Size(100){};

template <typename Scalar>
std::vector<cv::Point> Detector2D<Scalar>::ellipse_true_support( Ellipse& ellipse, Scalar ellipse_circumference, std::vector<cv::Point>& raw_edges){

  Scalar major_radius = ellipse.major_radius;
  Scalar minor_radius = ellipse.minor_radius;

  //std::cout << ellipse_circumference << std::endl;
  std::vector<Scalar> distances;
  std::vector<cv::Point> support_pixels;

  for(auto it = raw_edges.begin(); it != raw_edges.end(); it++){

      const cv::Point p = *it;
      Scalar distance = euclidean_distance( (Scalar)p.x, (Scalar)p.y, ellipse);  // change this one, to approxx ?
      std::cout << ellipse.center << std::endl;
      if(distance <=  1.3 ){
        support_pixels.push_back(p);
      }
  }

  return std::move(support_pixels);
}
template<typename Scalar>
Result<Scalar> Detector2D<Scalar>::detect( DetectProperties& props, cv::Mat& image, cv::Mat& color_image, cv::Mat& debug_image, cv::Rect& usr_roi , bool visualize, bool use_debug_image ){

  Result<Scalar> result;

  //remove this later
  debug_image = cv::Scalar(0); //clear debug image

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
  cv::inRange( image, cv::Scalar(0) , cv::Scalar(highest_spike_index - spectral_offset), spec_mask );  // binary threshold
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
  // if( mUse_strong_prior ){

  //   mUse_strong_prior = false;

  //   //recalculate center in coords system of ROI views
  //   Ellipse ellipse = mPrior_ellipse;
  //   ellipse.center[0] -= ( pupil_roi.x + usr_roi.x  );
  //   ellipse.center[1] -= ( pupil_roi.y + usr_roi.y  );

  //   if( !raw_edges.empty() ){

  //     std::vector<cv::Point> support_pixels;
  //     double ellipse_circumference = ellipse.circumference();
  //     support_pixels = ellipse_true_support(ellipse, ellipse_circumference, raw_edges);

  //     double support_ration = support_pixels.size() / ellipse_circumference;

  //     if(support_ration >= props.strong_perimeter_ratio_range_min){

  //       cv::RotatedRect refit_ellipse = cv::fitEllipse(support_pixels);

  //       if(use_debug_image){
  //           cv::ellipse(debug_image, toRotatedRect(ellipse), mRoyalBlue_color, 4);
  //           cv::ellipse(debug_image, refit_ellipse, mRed_color, 1);
  //       }

  //       ellipse = toEllipse<double>(refit_ellipse);
  //       ellipse.center[0] += ( pupil_roi.x + usr_roi.x  );
  //       ellipse.center[1] += ( pupil_roi.y + usr_roi.y  );
  //       mPrior_ellipse = ellipse;

  //       double goodness = std::min(0.1, support_ration);

  //       result.confidence = goodness;
  //       result.ellipse = ellipse;

  //       mPupil_Size = ellipse.major_radius;

  //       return result;

  //     }
  //   }
  // }

  ///////////////////////////////
  ///  Strong Prior Part End  ///
  ///////////////////////////////

  //from edges to contours
  std::vector<std::vector<cv::Point>> contours, good_contours;
  cv::findContours(edges, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
  good_contours.resize( contours.size() );

  //first we want to filter out the bad stuff, to short
  auto min_contour_size_pred = [&]( const std::vector<cv::Point>& contour){
    return contour.size() > props.contour_size_min;
  };
  auto end = std::copy_if( contours.begin(), contours.end(), good_contours.begin(), min_contour_size_pred); // better way than copy, erase probably not better with vector
  good_contours.resize(std::distance(good_contours.begin(),end));

  //now we learn things about each contour through looking at the curvature.
  //For this we need to simplyfy the contour so that pt to pt angles become more meaningfull

  std::vector<std::vector<cv::Point>> approx_contours;
  std::for_each(good_contours.begin(), good_contours.end(), [&]( std::vector<cv::Point>& contour){
    std::vector<cv::Point> approx_c;
    cv::approxPolyDP( contour, approx_c, 1.5, false);
    approx_contours.push_back(approx_c);
  });

  std::vector<std::vector<cv::Point>> split_contours;
  for(auto it = approx_contours.begin(); it != approx_contours.end(); it++ ){

    std::vector<cv::Point>& contour  = *it;
    std::vector<Scalar> curvature;
    // closed curves not handled yet
    for(auto point_it = contour.begin(); point_it != contour.end()-2; point_it++){
        cv::Point& first = *point_it;
        cv::Point& second = *(point_it+1);
        cv::Point& third = *(point_it+2);
        curvature.push_back( math::getAngleABC<Scalar>(first, second, third) );
    }

    //we split whenever there is a real kink (abs(curvature)<right angle) or a change in the genreal direction

    auto kink_indices = find_kink_and_dir_change( curvature, 80);
    auto contour_segments = split_at_corner_index( contour, kink_indices);
    //TODO: split at shart inward turns
    int colorIndex = 0;
    for( auto seg_it = contour_segments.begin(); seg_it != contour_segments.end(); seg_it++){

      std::vector<cv::Point> segment = *seg_it;
      //printPoints(segment);
      if( segment.size() > 2 ){
        split_contours.push_back(segment);

        // debug segments
        if(use_debug_image){
          const cv::Scalar_<int> colors[] = {mRed_color, mBlue_color, mRoyalBlue_color, mYellow_color, mWhite_color, mGreen_color};
          cv::polylines(debug_image, segment, false, colors[colorIndex], 1, 4);
          colorIndex++;
          colorIndex %= 6;
        }
      }
    }
  }

  if( split_contours.empty()){
    result.confidence = 0.0;
    return result;
  }

  //removing stubs makes combinatorial search feasable
  // erase-remove idiom

  split_contours.erase(
    std::remove_if(split_contours.begin(), split_contours.end(), [](std::vector<cv::Point>& v){ return v.size() <= 3;}), split_contours.end()
  );

  //finding poential candidates for ellipse seeds that describe the pupil.
  std::vector<int> strong_seed_contours;
  std::vector<int> weak_seed_contours;


  auto ellipse_filter = [&](const cv::RotatedRect& ellipse ) -> bool {
      bool is_centered = padding < ellipse.center.x  && ellipse.center.x < (image_width - padding) && padding < ellipse.center.y && ellipse.center.y < (image_height - padding);
      if(is_centered){

        float max_radius = ellipse.size.height;
        float min_radius = ellipse.size.width;
        if(min_radius > max_radius){
            std::cout << "Major Minor swizzle" << std::endl; // not happend yet, can we remove it ?
            min_radius = ellipse.size.height;
            max_radius = ellipse.size.width;
        }
        bool is_round = (min_radius/max_radius) >= props.ellipse_roundness_ratio;
        if(is_round){
            bool right_size = props.pupil_size_min <= max_radius && max_radius <= props.pupil_size_max;
            if(right_size) return true;
        }
      }
      return false;
  };


  for(int i=0; i < split_contours.size(); i++){

    auto contour = split_contours.at(i);

    if( contour.size() >= 5 ){ // because fitEllipse needs at least 5 points

      cv::RotatedRect ellipse = cv::fitEllipse(contour);
      //is this ellipse a plausible candidate for a pupil?
      if( ellipse_filter(ellipse) ){
        auto e = toEllipse<Scalar>(ellipse);
        Scalar point_distances = 0.0;
        EllipseDistCalculator<Scalar> ellipseDistance(e);

        //std::cout << "Ellipse: "  << ellipse.center  << " " << ellipse.size << " "<< ellipse.angle << std::endl;
        //std::cout << "Points: ";
        for(int j=0; j < contour.size(); j++){
            auto point = contour.at(j);
            //std::cout << point << ", ";
            point_distances += std::pow( std::abs( ellipseDistance( (Scalar)point.x, (Scalar)point.y )), 2 );
           // std::cout << "d=" << distance << ", " <<std::endl;
        }
       // std::cout << std::endl;

        Scalar fit_variance = point_distances / contour.size();
        //std::cout  << fit_variance <<std::endl;
        if( fit_variance < props.initial_ellipse_fit_treshhold ){
          // how much ellipse is supported by this contour?

          auto ellipse_contour_support_ratio = []( Ellipse& ellipse, std::vector<cv::Point>& contour ){

                Scalar ellipse_circumference = ellipse.circumference();
                Scalar ellipse_area = ellipse.area();
                std::vector<cv::Point> hull;
                cv::convexHull(contour, hull);
                Scalar actual_area = cv::contourArea(hull);
                Scalar actual_length  = cv::arcLength(contour, false);
                Scalar area_ratio = actual_area / ellipse_area;
                Scalar perimeter_ratio = actual_length / ellipse_circumference; //we assume here that the contour lies close to the ellipse boundary
                return std::pair<Scalar,Scalar>(area_ratio,perimeter_ratio);
          };

          auto ratio = ellipse_contour_support_ratio(e, contour);
          Scalar area_ratio = ratio.first;
          Scalar perimeter_ratio = ratio.second;
          //std::cout << area_ratio << ", " << perimeter_ratio << std::endl;
          if( props.strong_perimeter_ratio_range_min <= perimeter_ratio && perimeter_ratio <= props.strong_perimeter_ratio_range_max &&
              props.strong_area_ratio_range_min <= area_ratio && area_ratio <= props.strong_area_ratio_range_max ){

            strong_seed_contours.push_back(i);
            if(use_debug_image){
              cv::polylines( debug_image, contour, false, mRoyalBlue_color, 4);
              cv::ellipse( debug_image, ellipse, mBlue_color);
            }

          }else{
            weak_seed_contours.push_back(i);
            if(use_debug_image){
              cv::polylines( debug_image, contour, false, mBlue_color, 2);
              cv::ellipse( debug_image, ellipse, mBlue_color);
            }
          }
        }
      }
    }
  }

  std::vector<int>& seed_indices = strong_seed_contours;

  if( seed_indices.empty() && !weak_seed_contours.empty() ){
      seed_indices = weak_seed_contours;
  }else{
      result.confidence = 0.0;
      return result;
  }

//  std::cout << seed_indices.size() << std::endl;

  auto ellipse_evaluation = [&]( std::vector<cv::Point>& contour) -> bool {

      auto ellipse = cv::fitEllipse(contour);
      Scalar point_distances = 0.0;
      EllipseDistCalculator<Scalar> ellipseDistance( toEllipse<Scalar>(ellipse) );
      for(int i=0; i < contour.size(); i++){
          auto point = contour.at(i);
          point_distances += std::pow(std::abs( ellipseDistance( (Scalar)point.x, (Scalar)point.y )), 2);
      }
      Scalar fit_variance = point_distances / float(contour.size());
      std::cout << fit_variance << std::endl;
      return fit_variance <= props.initial_ellipse_fit_treshhold;

  };

  auto pruning_quick_combine = [&]( std::vector<std::vector<cv::Point>>& contours,  std::set<int>& seed_indices, int max_evals = 1e20, int max_depth = 5  ){

    typedef std::set<int> Path;

    std::vector<Path> unknown(seed_indices.size());
      // init with paths of size 1 == seed indices
    int n = 0;
    std::generate( unknown.begin(), unknown.end(), [&](){ return Path{n++}; }); // fill with increasing values, starting from 0

    std::vector<int> mapping(contours.size()); // contains all indices, starting with seed_indices
    mapping.insert(mapping.end(), seed_indices.begin(), seed_indices.end());
    // add indices which are not used to the end of mapping
    for( int i=0; i < contours.size(); i++){
      if( seed_indices.find(i) != seed_indices.end() ){ mapping.push_back(i); }
    }

    // contains all the indices for the contours, which altogther fit best
    std::vector<Path> results;
    // contains bad paths
    std::vector<Path> prune;

    int eval_count = 0;
    while( !unknown.empty() && eval_count <= max_evals ){

      eval_count++;
      //take a path and combine it with others to see if the fit gets better
      Path current_path = unknown.back();
      unknown.pop_back();
      if( current_path.size() <= max_depth ){

          bool includes_bad_paths = false;
          for( Path& bad_path: prune){
            // check if bad_path is a subset of current_path
            // for std::include both containers need to be ordered. std::set guarantees this
            includes_bad_paths |= std::includes(current_path.begin(), current_path.end(), bad_path.begin(), bad_path.end());
          }

          if( !includes_bad_paths ){
              int size = 0;
              for( int i : current_path ){ size += contours.at(mapping.at(i)).size(); };
              std::vector<cv::Point> test_contour(size); // reserve size
              std::set<int> test_contour_indices;
              //concatenate contours to one contour
              for( int i : current_path ){
               std::vector<cv::Point>& c = contours.at(mapping.at(i));
               test_contour.insert( test_contour.end(), c.begin(), c.end() );
               test_contour_indices.insert(i);
              }
             // std::cout << "evaluate ellipse " << std::endl;
              std::cout << "amount contours: " << current_path.size() << std::endl;

              //we have not tested this and a subset of this was sucessfull before
              if( ellipse_evaluation( test_contour ) ){

                //yes this was good, keep as solution
                results.push_back( test_contour_indices );
               // std::cout << "add result" << std::endl;
                //lets explore more by creating paths to each remaining node
                for(int j= (*current_path.rbegin())+1 ; j < mapping.size(); j++  ){
                    unknown.push_back( current_path );
                    unknown.back().insert(j); // add a new path
                }

              }else{
                prune.push_back( current_path);
              }
          }
      }
    }
    return results;
  };

  std::set<int> seed_indices_set = std::set<int>(seed_indices.begin(),seed_indices.end());
  std::vector<std::set<int>> solutions = pruning_quick_combine( split_contours, seed_indices_set, 1000, 5);

 // std::cout << "solutions: " << solutions.size() << std::endl;

  //find largest sets which contains all previous ones
  auto filter_subset = [](std::vector<std::set<int>>& sets){
    std::vector<std::set<int>> filtered_set;
    int i = 0;
    for(auto& current_set : sets){

        //check if this current_set is a subset of set
        bool isSubset = false;
        for( int j = 0; j < sets.size(); j++){
          //if(j == i ) continue;// don't compare to itself
          auto& set = sets.at(j);
          // for std::include both containers need to be ordered. std::set guarantees this
          isSubset |= std::includes(set.begin(), set.end(), current_set.begin(), current_set.end());
        }

        if(!isSubset){
           filtered_set.push_back(current_set);
        }
        i++;
    }
    return filtered_set;
  };

  solutions = filter_subset(solutions);

  // for( auto& s : solutions){

  //   std::vector<cv::Point> test_contour;
  //   //concatenate contours to one contour
  //   for( int i : s ){
  //    std::vector<cv::Point>& c = split_contours.at(i);
  //    test_contour.insert( test_contour.end(), c.begin(), c.end() );
  //   }
  //   auto ellipse = cv::fitEllipse( test_contour );

  //   if(use_debug_image ){
  //       std::cout << "debug "<< std::endl;
  //       cv::ellipse(debug_image, ellipse , mYellow_color);
  //   }
  // }

  //cv::drawContours(debug_image, approx_contours, -1, mGreen_color, 2);
  cv::imshow("debug_image", debug_image);

  return result;

}
