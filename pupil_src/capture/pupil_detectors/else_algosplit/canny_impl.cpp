


/*
  Version 1.0, 08.06.2015, Copyright University of Tübingen.

  The Code is created based on the method from the paper:
  "ExCuSe: Robust Pupil Detection in Real-World Scenarios", W. Fuhl, T. C. Kübler, K. Sippel, W. Rosenstiel, E. Kasneci
  CAIP 2015 : Computer Analysis of Images and Patterns

  The code and the algorithm are for non-comercial use only.

*/



#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>


namespace ELSE{


static float hypot(float a,float b)
{
	a=abs(a);
	b=abs(b);
    float t=a<b?a:b;
	float x=a>b?a:b;

	if(x==0)
		return 0;


    t = t/x;
    return x*sqrt(1+(t*t));
}






#define MAX_LINE 100000

static void matlab_bwselect(cv::Mat *strong,cv::Mat *weak,cv::Mat *check){

	int pic_x=strong->cols;
	int pic_y=strong->rows;

	int lines[MAX_LINE];
	int lines_idx=0;


	int idx=0;

	char *p_strong, *p_weak, *p_check;
	for(int i=1;i<pic_y-1;i++){

		for(int j=1;j<pic_x-1;j++){

			if(strong->data[idx+j]!=0 && check->data[idx+j]==0){

				check->data[idx+j]=255;
				lines_idx=1;
				lines[0]=idx+j;


				int akt_idx=0;

				while(akt_idx<lines_idx && lines_idx<MAX_LINE-1){

					int akt_pos=lines[akt_idx];

					if(akt_pos-pic_x-1>=0 && akt_pos+pic_x+1<pic_x*pic_y){
					for(int k1=-1;k1<2;k1++)
						for(int k2=-1;k2<2;k2++){


							if(check->data[(akt_pos+(k1*pic_x))+k2]==0 && weak->data[(akt_pos+(k1*pic_x))+k2]!=0){

								check->data[(akt_pos+(k1*pic_x))+k2]=255;

								lines_idx++;
								lines[lines_idx-1]=(akt_pos+(k1*pic_x))+k2;
							}

					}
					}
					akt_idx++;

				}





			}

		}

	idx+=pic_x;
	}

}





static cv::Mat canny_impl(cv::Mat *pic, cv::Mat *magni){
	int k_sz=16;


	float gau[16] = {0.000000220358050,0.000007297256405,0.000146569312970,0.001785579770079,
						0.013193749090229,0.059130281094460,0.160732768610747,0.265003534507060,0.265003534507060,
						0.160732768610747,0.059130281094460,0.013193749090229,0.001785579770079,0.000146569312970,
						0.000007297256405,0.000000220358050};
	float deriv_gau[16] = {-0.000026704586264,-0.000276122963398,-0.003355163265098,-0.024616683775044,-0.108194751875585,
								-0.278368310241814,-0.388430056419619,-0.196732206873178,0.196732206873178,0.388430056419619,
								0.278368310241814,0.108194751875585,0.024616683775044,0.003355163265098,0.000276122963398,0.000026704586264};





	cv::Point anchor = cv::Point( -1, -1 );
	float delta = 0;
	int ddepth = -1;


	pic->convertTo(*pic, CV_32FC1);


	cv::Mat gau_x = cv::Mat(1, k_sz, CV_32FC1,&gau);
	cv::Mat deriv_gau_x = cv::Mat(1, k_sz, CV_32FC1,&deriv_gau);









	cv::Mat res_x;
	cv::Mat res_y;




	cv::transpose(*pic,*pic);
	filter2D(*pic, res_x, ddepth , gau_x, anchor, delta, cv::BORDER_REPLICATE );
	cv::transpose(*pic,*pic);
	cv::transpose(res_x,res_x);


	filter2D(res_x, res_x, ddepth , deriv_gau_x, anchor, delta, cv::BORDER_REPLICATE );



	filter2D(*pic, res_y, ddepth , gau_x, anchor, delta, cv::BORDER_REPLICATE );


	cv::transpose(res_y,res_y);
	filter2D(res_y, res_y, ddepth , deriv_gau_x, anchor, delta, cv::BORDER_REPLICATE );
	cv::transpose(res_y,res_y);







	*magni=cv::Mat::zeros(pic->rows, pic->cols, CV_32FC1);




	float * p_res, *p_x, *p_y;
	for(int i=0; i<magni->rows; i++){
		p_res=magni->ptr<float>(i);
		p_x=res_x.ptr<float>(i);
		p_y=res_y.ptr<float>(i);

		for(int j=0; j<magni->cols; j++){
			//res.at<float>(j, i)= sqrt( (res_x.at<float>(j, i)*res_x.at<float>(j, i)) + (res_y.at<float>(j, i)*res_y.at<float>(j, i)) );
			//res.at<float>(j, i)=robust_pytagoras_after_MOLAR_MORRIS(res_x.at<float>(j, i), res_y.at<float>(j, i));
			//res.at<float>(j, i)=hypot(res_x.at<float>(j, i), res_y.at<float>(j, i));

			//p_res[j]=__ieee754_hypot(p_x[j], p_y[j]);

			p_res[j]=hypot(p_x[j], p_y[j]);
		}
	}



	//th selection

	int PercentOfPixelsNotEdges=0.7 * magni->cols * magni->rows;
	float ThresholdRatio=0.4;

	float high_th=0;
	float low_th=0;

	int h_sz=64;
	int hist[64];
	for(int i=0; i<h_sz; i++) hist[i]=0;


	cv::normalize(*magni, *magni, 0, 1, cv::NORM_MINMAX, CV_32FC1);


	cv::Mat res_idx=cv::Mat::zeros(pic->rows, pic->cols, CV_8U);
	cv::normalize(*magni, res_idx, 0, 63, cv::NORM_MINMAX, CV_32S);

	int *p_res_idx=0;
	for(int i=0; i<magni->rows; i++){
		p_res_idx=res_idx.ptr<int>(i);
		for(int j=0; j<magni->cols; j++){
			hist[p_res_idx[j]]++;
	}}



	int sum=0;
	for(int i=0; i<h_sz; i++){
		sum+=hist[i];
		if(sum>PercentOfPixelsNotEdges){
			high_th=float(i+1)/float(h_sz);
			break;
		}
	}
	low_th=ThresholdRatio*high_th;





	//non maximum supression + interpolation
	cv::Mat non_ms=cv::Mat::zeros(pic->rows, pic->cols, CV_8U);
	cv::Mat non_ms_hth=cv::Mat::zeros(pic->rows, pic->cols, CV_8U);


	float ix,iy, grad1, grad2, d;

	char *p_non_ms,*p_non_ms_hth;
	float * p_res_t, *p_res_b;
	for(int i=1; i<magni->rows-1; i++){
		p_non_ms=non_ms.ptr<char>(i);
		p_non_ms_hth=non_ms_hth.ptr<char>(i);

		p_res=magni->ptr<float>(i);
		p_res_t=magni->ptr<float>(i-1);
		p_res_b=magni->ptr<float>(i+1);

		p_x=res_x.ptr<float>(i);
		p_y=res_y.ptr<float>(i);


		for(int j=1; j<magni->cols-1; j++){


				iy=p_y[j];
				ix=p_x[j];

				if( (iy<=0 && ix>-iy) || (iy>=0 && ix<-iy) ){


					d=abs(iy/ix);
					grad1=( p_res[j+1]*(1-d) ) + ( p_res_t[j+1]*d );
					grad2=( p_res[j-1]*(1-d) ) + ( p_res_b[j-1]*d );

					if(p_res[j]>=grad1 && p_res[j]>=grad2){
						p_non_ms[j]=255;

						if(p_res[j]>high_th)
							p_non_ms_hth[j]=255;
					}
				}





				if( (ix>0 && -iy>=ix)  || (ix<0 && -iy<=ix) ){
					d=abs(ix/iy);
					grad1=( p_res_t[j]*(1-d) ) + ( p_res_t[j+1]*d );
					grad2=( p_res_b[j]*(1-d) ) + ( p_res_b[j-1]*d );

					if(p_res[j]>=grad1 && p_res[j]>=grad2){
						p_non_ms[j]=255;
						if(p_res[j]>high_th)
							p_non_ms_hth[j]=255;
					}
				}



				if( (ix<=0 && ix>iy) || (ix>=0 && ix<iy) ){
					d=abs(ix/iy);
					grad1=( p_res_t[j]*(1-d) ) + ( p_res_t[j-1]*d );
					grad2=( p_res_b[j]*(1-d) ) + ( p_res_b[j+1]*d );

					if(p_res[j]>=grad1 && p_res[j]>=grad2){
						p_non_ms[j]=255;
						if(p_res[j]>high_th)
							p_non_ms_hth[j]=255;
					}
				}



				if( (iy<0 && ix<=iy) || (iy>0 && ix>=iy)){
					d=abs(iy/ix);
					grad1=( p_res[j-1]*(1-d) ) + ( p_res_t[j-1]*d );
					grad2=( p_res[j+1]*(1-d) ) + ( p_res_b[j+1]*d );

					if(p_res[j]>=grad1 && p_res[j]>=grad2){
						p_non_ms[j]=255;
						if(p_res[j]>high_th)
							p_non_ms_hth[j]=255;
					}
				}

		}}






	////bw select
	cv::Mat res_lin=cv::Mat::zeros(pic->rows, pic->cols, CV_8U);
	matlab_bwselect(&non_ms_hth, &non_ms,&res_lin);




	pic->convertTo(*pic, CV_8U);




	return res_lin;

}



}