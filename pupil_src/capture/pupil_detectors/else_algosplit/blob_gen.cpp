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


static void mum(cv::Mat *pic, cv::Mat *result, int fak){


	int fak_ges=fak+1;
	int sz_x=pic->cols/fak_ges;
	int sz_y=pic->rows/fak_ges;

	*result=cv::Mat::zeros(sz_y, sz_x, CV_8U);

	int hist[256];
	int mean=0;
	int cnt=0;
	int mean_2=0;

	int idx=0;
	int idy=0;

	for(int i=0;i<sz_y;i++){
		idy+=fak_ges;

		for(int j=0;j<sz_x;j++){
			idx+=fak_ges;

			for(int k=0;k<256;k++)
				hist[k]=0;


			mean=0;
			cnt=0;


			for(int ii=-fak;ii<=fak;ii++)
				for(int jj=-fak;jj<=fak;jj++){

					if(idy+ii>0 && idy+ii<pic->rows && idx+jj>0 && idx+jj<pic->cols){
						if((unsigned int)pic->data[(pic->cols*(idy+ii))+(idx+jj)]>255)
							pic->data[(pic->cols*(idy+ii))+(idx+jj)]=255;

						hist[pic->data[(pic->cols*(idy+ii))+(idx+jj)]]++;
						cnt++;
						mean+=pic->data[(pic->cols*(idy+ii))+(idx+jj)];
					}


				}


			mean=mean/cnt;

			mean_2=0;
			cnt=0;
			for(int ii=0;ii<=mean;ii++){
				mean_2+=ii*hist[ii];
				cnt+=hist[ii];
			}

			if(cnt==0)
				mean_2=mean;
			else
				mean_2=mean_2/cnt;

			result->data[(sz_x*(i))+(j)]=mean_2;
		}

		idx=0;
	}


}

static void gen_blob_neu(int rad, cv::Mat *all_mat, cv::Mat *all_mat_neg){


		int len=1+(4*rad);
		int c0=rad*2;
		float negis=0;
		float posis=0;

		*all_mat = cv::Mat::zeros(len, len, CV_32FC1);
		*all_mat_neg = cv::Mat::zeros(len, len, CV_32FC1);


		float *p, *p_neg;
		for(int i=-rad*2;i<=rad*2;i++){ //height
			p=all_mat->ptr<float>(c0+i);

			for(int j=-rad*2;j<=rad*2;j++){

				if(i<-rad || i>rad){ //pos
					p[c0+j]=1;
					posis++;

				}else{ //neg

					int sz_w=sqrt( float(rad*rad) - float(i*i) );

					if(abs(j)<=sz_w){
						p[c0+j]=-1;
						negis++;
					}else{
						p[c0+j]=1;
						posis++;
					}

				}

			}
		}



	
	for(int i=0;i<len;i++){
		p=all_mat->ptr<float>(i);
		p_neg=all_mat_neg->ptr<float>(i);

		for(int j=0;j<len;j++){

			if(p[j]>0){
				p[j]=1.0/posis;
				p_neg[j]=0.0;
			}else{
				p[j]=-1.0/negis;
				p_neg[j]=1.0/negis;
			}

		}
	}



}





static bool is_good_ellipse_evaluation(cv::RotatedRect *ellipse, cv::Mat *pic){



	if(ellipse->center.x==0 && ellipse->center.y==0)
		return false;

	
	float x0 =ellipse->center.x;
	float y0 =ellipse->center.y;


	int st_x=x0-(ellipse->size.width/4.0);
	int st_y=y0-(ellipse->size.height/4.0);
	int en_x=x0+(ellipse->size.width/4.0);
	int en_y=y0+(ellipse->size.height/4.0);


	float val=0.0;
	float val_cnt=0;
	float ext_val=0.0;

	for(int i=st_x; i<en_x;i++)
		for(int j=st_y; j<en_y;j++){

			if(i>0 && i<pic->cols && j>0 && j<pic->rows ){
				val+=pic->data[(pic->cols*j)+i];
				val_cnt++;
			}
		}

	if(val_cnt>0)
		val=val/val_cnt;
	else
		return false;


	val_cnt=0;

	st_x=x0-(ellipse->size.width*0.75);
	st_y=y0-(ellipse->size.height*0.75);
	en_x=x0+(ellipse->size.width*0.75);
	en_y=y0+(ellipse->size.height*0.75);

	int in_st_x=x0-(ellipse->size.width/2);
	int in_st_y=y0-(ellipse->size.height/2);
	int in_en_x=x0+(ellipse->size.width/2);
	int in_en_y=y0+(ellipse->size.height/2);



	for(int i=st_x; i<en_x;i++)
		for(int j=st_y; j<en_y;j++){
			float x1=x0+i;
			float y1=y0+j;

			if(!(i>=in_st_x && i<=in_en_x && j>=in_st_y && j<=in_en_y))
			if(i>0 && i<pic->cols && j>0 && j<pic->rows ){
				ext_val+=pic->data[(pic->cols*j)+i];
				val_cnt++;
			}
		}




	if(val_cnt>0)
		ext_val=ext_val/val_cnt;
	else
		return false;



	val=ext_val-val;

	if(val>10) return true;
	else return false;
}




static cv::RotatedRect blob_finder(cv::Mat *pic){

	cv::Point pos(0,0);
	float abs_max=0;

	float *p_erg;
	cv::Mat blob_mat, blob_mat_neg;


	int fak_mum=5;
	int fakk=pic->cols>pic->rows?(pic->cols/100)+1:(pic->rows/100)+1;

	cv::Mat img;
	mum(pic, &img, fak_mum);
	cv::Mat erg = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);




	cv::Mat result, result_neg;


	gen_blob_neu(fakk,&blob_mat,&blob_mat_neg);
	
	img.convertTo(img, CV_32FC1);
	filter2D(img, result, -1 , blob_mat, cv::Point( -1, -1 ), 0, cv::BORDER_REPLICATE );

	
	float * p_res, *p_neg_res;
	for(int i=0; i<result.rows;i++){
		p_res=result.ptr<float>(i);

		for(int j=0; j<result.cols;j++){
			if(p_res[j]<0)
				p_res[j]=0;
		}
	}

	filter2D(img, result_neg, -1 , blob_mat_neg, cv::Point( -1, -1 ), 0, cv::BORDER_REPLICATE );
	

	for(int i=0; i<result.rows;i++){
		p_res=result.ptr<float>(i);
		p_neg_res=result_neg.ptr<float>(i);
		p_erg=erg.ptr<float>(i);

		for(int j=0; j<result.cols;j++){
				p_neg_res[j]=(255.0f-p_neg_res[j]);
				p_erg[j]=(p_neg_res[j])*(p_res[j]);
		}
	}







	float * p_img;
	for(int i=0; i<erg.rows;i++){
		p_erg=erg.ptr<float>(i);

		for(int j=0; j<erg.cols;j++){
			if(abs_max<p_erg[j]){
				abs_max=p_erg[j];


				pos.x=(fak_mum+1)+(j*(fak_mum+1));
				pos.y=(fak_mum+1)+(i*(fak_mum+1));

			}
		}
	}


if(pos.y>0 && pos.y<pic->rows && pos.x>0 && pos.x<pic->cols){
	
	//calc th
	int opti_x=0;
	int opti_y=0;

	float mm=0;
	float cnt=0;
	for(int i=-(2); i<(2);i++){
		for(int j=-(2); j<(2);j++){
			if( pos.y+i>0 && pos.y+i<pic->rows && pos.x+j>0 && pos.x+j<pic->cols){
				mm+=pic->data[(pic->cols*(pos.y+i))+(pos.x+j)];
				cnt++;
			}

		}
	}

	if(cnt>0)
		mm=ceil(mm/cnt);


	int th_bot=0;
	if(pos.y>0 && pos.y<pic->rows && pos.x>0 && pos.x<pic->cols)
		th_bot= pic->data[(pic->cols*(pos.y))+(pos.x)] + abs(mm-pic->data[(pic->cols*(pos.y))+(pos.x)]);
	cnt=0;

	for(int i=-(fak_mum*fak_mum); i<(fak_mum*fak_mum);i++){
		for(int j=-(fak_mum*fak_mum); j<(fak_mum*fak_mum);j++){

			if( pos.y+i>0 && pos.y+i<pic->rows && pos.x+j>0 && pos.x+j<pic->cols){

				if(pic->data[(pic->cols*(pos.y+i))+(pos.x+j)]<=th_bot){
					opti_x+=pos.x+j;
					opti_y+=pos.y+i;
					cnt++;
				}
			}

		}
	}


	if(cnt>0){
		opti_x=opti_x/cnt;
		opti_y=opti_y/cnt;
	}else{
		opti_x=pos.x;
		opti_y=pos.y;
	}

	pos.x=opti_x;
	pos.y=opti_y;



}


	cv::RotatedRect ellipse;

	if( pos.y>0 && pos.y<pic->rows && pos.x>0 && pos.x<pic->cols){
		ellipse.center.x=pos.x;
		ellipse.center.y=pos.y;
		ellipse.angle=0.0;
		ellipse.size.height=(fak_mum*fak_mum*2) +1;
		ellipse.size.width=(fak_mum*fak_mum*2) +1;

		if(!is_good_ellipse_evaluation(&ellipse, pic)){
			ellipse.center.x=0;
			ellipse.center.y=0;
			ellipse.angle=0;
			ellipse.size.height=0;
			ellipse.size.width=0;
		}

	}else{
		ellipse.center.x=0;
		ellipse.center.y=0;
		ellipse.angle=0;
		ellipse.size.height=0;
		ellipse.size.width=0;

	}



	
	return ellipse;

}

}