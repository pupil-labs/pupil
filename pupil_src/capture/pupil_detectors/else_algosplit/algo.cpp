/*
  Version 1.0, 17.12.2015, Copyright University of Tübingen.

  The Code is created based on the method from the paper:
  "ElSe: Ellipse Selection for Robust Pupil Detection in Real-World Environments", W. Fuhl, T. C. Santini, T. C. Kübler, E. Kasneci
  ETRA 2016 : Eye Tracking Research and Application 2016

  The code and the algorithm are for non-comercial use only.

*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>



#include "find_best_edge.cpp"
#include "canny_impl.cpp"
#include "blob_gen.cpp"
#include "filter_edges.cpp"


namespace ELSE{


static cv::RotatedRect run(cv::Mat input_img){

	float rz_fakk=float(input_img.cols)/384.0;

	cv::Mat pic=cv::Mat::zeros(input_img.rows/rz_fakk, input_img.cols/rz_fakk, CV_8UC1);
	cv::resize(input_img, pic,pic.size());
	cv::normalize(pic, pic, 0, 255, cv::NORM_MINMAX, CV_8U);

	double border=0.05;//0.1
	double mean_dist=3;
	int inner_color_range=0;

	cv::RotatedRect ellipse;
	cv::Point pos(0,0);

	int start_x=floor(double(pic.cols)*border);
	int start_y=floor(double(pic.rows)*border);

	int end_x =pic.cols-start_x;
	int end_y =pic.rows-start_y;

	cv::Mat picpic = cv::Mat::zeros(end_y-start_y, end_x-start_x, CV_8U);
	cv::Mat magni;

	for(int i=0; i<picpic.cols; i++)
		for(int j=0; j<picpic.rows; j++){
			picpic.data[(picpic.cols*j)+i]=pic.data[(pic.cols*(start_y+j))+(start_x+i)];
		}

	cv::Mat detected_edges2 = canny_impl(&picpic,&magni);

	cv::Mat detected_edges = cv::Mat::zeros(pic.rows, pic.cols, CV_8U);
	for(int i=0; i<detected_edges2.cols; i++)
		for(int j=0; j<detected_edges2.rows; j++){
			detected_edges.data[(detected_edges.cols*(start_y+j))+(start_x+i)]=detected_edges2.data[(detected_edges2.cols*j)+i];
		}

	filter_edges(&detected_edges, start_x, end_x, start_y, end_y);

	ellipse=find_best_edge(&pic, &detected_edges, &magni, start_x, end_x, start_y, end_y,mean_dist, inner_color_range);

	if(ellipse.center.x<=0 && ellipse.center.y<=0 || ellipse.center.x>=pic.cols || ellipse.center.y>=pic.rows){

		ellipse=blob_finder(&pic);

	}

	ellipse.size.height=ellipse.size.height*rz_fakk;
	ellipse.size.width=ellipse.size.width*rz_fakk;


	ellipse.center.x=ellipse.center.x*rz_fakk;
	ellipse.center.y=ellipse.center.y*rz_fakk;

	return ellipse;




}




}



