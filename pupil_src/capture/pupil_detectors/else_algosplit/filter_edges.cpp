#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

/*
  Version 1.0, 17.12.2015, Copyright University of Tübingen.

  The Code is created based on the method from the paper:
  "ElSe: Ellipse Selection for Robust Pupil Detection in Real-World Environments", W. Fuhl, T. C. Santini, T. C. Kübler, E. Kasneci
  ETRA 2016 : Eye Tracking Research and Application 2016
 
  The code and the algorithm are for non-comercial use only.

*/

namespace ELSE{


static void filter_edges(cv::Mat *edge, int start_xx, int end_xx, int start_yy, int end_yy){



		int start_x=start_xx+5;
		int end_x=end_xx-5;
		int start_y=start_yy+5;
		int end_y=end_yy-5;


		if(start_x<5) start_x=5;
		if(end_x>edge->cols-5) end_x=edge->cols-5;
		if(start_y<5) start_y=5;
		if(end_y>edge->rows-5) end_y=edge->rows-5;




		for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){
			int box[9];

			box[4]=(int)edge->data[(edge->cols*(j))+(i)];

			if(box[4]){
				box[1]=(int)edge->data[(edge->cols*(j-1))+(i)];
				box[3]=(int)edge->data[(edge->cols*(j))+(i-1)];
				box[5]=(int)edge->data[(edge->cols*(j))+(i+1)];
				box[7]=(int)edge->data[(edge->cols*(j+1))+(i)];


				if((box[5] && box[7])) edge->data[(edge->cols*(j))+(i)]=0;
				if((box[5] && box[1])) edge->data[(edge->cols*(j))+(i)]=0;
				if((box[3] && box[7])) edge->data[(edge->cols*(j))+(i)]=0;
				if((box[3] && box[1])) edge->data[(edge->cols*(j))+(i)]=0;

			}
		}



		//too many neigbours
		for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){
			int neig=0;

			for(int k1=-1;k1<2;k1++)
				for(int k2=-1;k2<2;k2++){

					if(edge->data[(edge->cols*(j+k1))+(i+k2)]>0)
						neig++;
				}

			if(neig>3)
				edge->data[(edge->cols*(j))+(i)]=0;

		}








}


}
