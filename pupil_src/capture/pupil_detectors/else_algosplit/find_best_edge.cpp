


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

namespace ELSE{

#define IMG_SIZE 1000 //400


static float calc_angle(cv::Point vec, cv::Point ortho){

	float norm=sqrt( double((vec.x*vec.x)+(vec.y*vec.y)) ) * sqrt( double((ortho.x*ortho.x)+(ortho.y*ortho.y)) );
	float dot_prod= (vec.x*ortho.x)+(vec.y*ortho.y);

	if(norm==0)
		return 0;
	else
		return acos(dot_prod/norm)*57.2957795;


}




static bool is_good_ellipse_eval(cv::RotatedRect *ellipse, cv::Mat *pic, int *erg){



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


	*erg=val;

	if(val>10) return true;
	else return false;
}





static int calc_inner_gray(cv::Mat *pic, std::vector<cv::Point> curve, cv::RotatedRect ellipse){

	int gray_val=0;
	int gray_cnt=0;



	cv::Mat checkmap = cv::Mat::zeros(pic->size(),CV_8U);


	for(int i=0;i<curve.size();i++){

		int vec_x=curve[i].x-ellipse.center.x;
		int vec_y=curve[i].y-ellipse.center.y;

		for(float p=0.95;p>0.80;p-=0.01){//0.75;-0.05
			int p_x=ellipse.center.x+float((float(vec_x)*p)+0.5);
			int p_y=ellipse.center.y+float((float(vec_y)*p)+0.5);


			if(p_x>0 && p_x<pic->cols && p_y>0 && p_y<pic->rows){

				if(checkmap.data[(pic->cols*p_y)+p_x]==0){
					checkmap.data[(pic->cols*p_y)+p_x]=1;
					gray_val+=(unsigned int)pic->data[(pic->cols*p_y)+p_x];
					gray_cnt++;
				}

			}


		}


	}


	if(gray_cnt>0)
		gray_val=gray_val/gray_cnt;
	else
		gray_val=1000;

	return gray_val;
}




static std::vector<std::vector<cv::Point>> get_curves(cv::Mat *pic, cv::Mat *edge, cv::Mat *magni, int start_x, int end_x, int start_y, int end_y, double mean_dist, int inner_color_range){

	std::vector<std::vector<cv::Point>> all_lines;
	int size_open=0;

	std::vector<std::vector<cv::Point>> all_curves;
	std::vector<cv::Point> curve;

	std::vector<cv::Point> all_means;


	if(start_x<2) start_x=2;
	if(start_y<2) start_y=2;
	if(end_x>pic->cols-2) end_x=pic->cols-2;
	if(end_y>pic->rows-2) end_y=pic->rows-2;

	
	int all_curves_idx=0;
	int curve_idx=0;
	cv::Point mean_p;
	bool add_curve;
	int mean_inner_gray;
	int mean_inner_gray_last=1000000;
	

	all_curves.clear();
	all_means.clear();
	all_lines.clear();



	bool check[IMG_SIZE][IMG_SIZE];

	for(int i=0; i<IMG_SIZE; i++)
		for(int j=0; j<IMG_SIZE; j++)
			check[i][j]=0;




		
	//get all open lines 
	for(int i=start_x; i<end_x; i++)
		for(int j=start_y; j<end_y; j++){

			
			

			if(edge->data[(edge->cols*(j))+(i)]>0 && !check[i][j]){




				int neig=0;

				for(int k1=-1;k1<2;k1++)
				for(int k2=-1;k2<2;k2++){

					if(edge->data[(edge->cols*(j+k1))+(i+k2)]>0)
						neig++;
				}

				if(neig==2){




				check[i][j]=1;
				
				curve.clear();
				curve_idx=0;

				curve.push_back(cv::Point(i,j));
				mean_p.x=i;
				mean_p.y=j;
				curve_idx++;

				
				int akt_idx=0;

				while(akt_idx<curve_idx){

					cv::Point akt_pos=curve[akt_idx];
					for(int k1=-1;k1<2;k1++)
						for(int k2=-1;k2<2;k2++){

							if(akt_pos.x+k1>=start_x && akt_pos.x+k1<end_x && akt_pos.y+k2>=start_y && akt_pos.y+k2<end_y)
							if(!check[akt_pos.x+k1][akt_pos.y+k2] )
								if( edge->data[(edge->cols*(akt_pos.y+k2))+(akt_pos.x+k1)]>0){
								check[akt_pos.x+k1][akt_pos.y+k2]=1;
								
								mean_p.x+=akt_pos.x+k1;
								mean_p.y+=akt_pos.y+k2;
								curve.push_back(cv::Point(akt_pos.x+k1,akt_pos.y+k2));
								curve_idx++;
							}
							
					}
					akt_idx++;

				}





				if(curve_idx>5 && curve.size()>5){

				

					
					mean_p.x=floor(( double(mean_p.x)/double(curve_idx) )+0.5);
					mean_p.y=floor(( double(mean_p.y)/double(curve_idx) )+0.5);


					all_means.push_back(mean_p);
					all_lines.push_back(curve);


				}



			}
			}
		}



	//mark size of closed lines
		size_open=all_lines.size();


		

	//get all closed lines 
		for(int i=start_x; i<end_x; i++)
		for(int j=start_y; j<end_y; j++){

			
			

			if(edge->data[(edge->cols*(j))+(i)]>0 && !check[i][j]){
				check[i][j]=1;
				
				curve.clear();
				curve_idx=0;

				curve.push_back(cv::Point(i,j));
				mean_p.x=i;
				mean_p.y=j;
				curve_idx++;

				
				int akt_idx=0;

				while(akt_idx<curve_idx){

					cv::Point akt_pos=curve[akt_idx];
					for(int k1=-1;k1<2;k1++)
						for(int k2=-1;k2<2;k2++){

							if(akt_pos.x+k1>=start_x && akt_pos.x+k1<end_x && akt_pos.y+k2>=start_y && akt_pos.y+k2<end_y)
							if(!check[akt_pos.x+k1][akt_pos.y+k2] )
								if( edge->data[(edge->cols*(akt_pos.y+k2))+(akt_pos.x+k1)]>0){
								check[akt_pos.x+k1][akt_pos.y+k2]=1;
								
								mean_p.x+=akt_pos.x+k1;
								mean_p.y+=akt_pos.y+k2;
								curve.push_back(cv::Point(akt_pos.x+k1,akt_pos.y+k2));
								curve_idx++;
							}
							
					}
					akt_idx++;

				}






				if(curve_idx>10 && curve.size()>10){

					mean_p.x=floor(( double(mean_p.x)/double(curve_idx) )+0.5);
					mean_p.y=floor(( double(mean_p.y)/double(curve_idx) )+0.5);

					all_means.push_back(mean_p);
					all_lines.push_back(curve);
					


				}

			}
		}




	//break up lines (roundness check)
	for(int iii=0;iii<size_open;iii++) {
		curve.clear();
		curve=all_lines.at(iii);

		std::vector<cv::Point> new_curve;
		new_curve.clear();

		cv::Point ortho,first,secound, ak_vec;

		ortho.x=0;first.x=0;secound.x=0;ak_vec.x=0;
		ortho.y=0;first.y=0;secound.y=0;ak_vec.y=0;
		

		float main_angle=-1000;
		float main_dist=-1000;
		float ak_angle,ak_dist;
		bool change=false;


		//for(int k=curve.size()-1;k>-1;k--){
		for(int k=0;k<curve.size();k++){

			if(first.x==0 && first.y==0){
				new_curve.clear();
				change=false;
				main_dist=1;

				new_curve.push_back(curve.at(k));
				mean_p.x=curve.at(k).x;
				mean_p.y=curve.at(k).y;

				first.x=curve.at(k).x;
				first.y=curve.at(k).y;
			}else if(secound.x==0 && secound.y==0){

				new_curve.push_back(curve.at(k));
				mean_p.x+=curve.at(k).x;
				mean_p.y+=curve.at(k).y;

				secound.x=curve.at(k).x;
				secound.y=curve.at(k).y;
			}else{



			if(ortho.x==0 && ortho.y==0){
				ortho.x=secound.y-first.y;
				ortho.y=-(secound.x-first.x);
			}


			ak_vec.x=curve.at(k).x-first.x;
			ak_vec.y=curve.at(k).y-first.y;
			ak_angle=calc_angle(ak_vec, ortho);
			ak_dist=sqrt(double(ak_vec.x*ak_vec.x)+double(ak_vec.y*ak_vec.y));

			if(main_angle==-1000){
				

				if(ak_angle>=90){
					ortho.x=-ortho.x;
					ortho.y=-ortho.y;
					ak_angle=calc_angle(ak_vec, ortho);
				}

				if(ak_angle>=90){
					//not a curve
					ortho.x=0;first.x=0;secound.x=0;ak_vec.x=0;
					ortho.y=0;first.y=0;secound.y=0;ak_vec.y=0;
					main_angle=-1000;
				}else{
					main_angle=ak_angle;
					new_curve.push_back(curve.at(k));
					mean_p.x+=curve.at(k).x;
					mean_p.y+=curve.at(k).y;
				}


			}else{



				if(!change){
				if(main_angle>=ak_angle && ak_angle<=90 && main_dist<=ak_dist){
					main_angle=ak_angle;
					main_dist=ak_dist;
					new_curve.push_back(curve.at(k));
					mean_p.x+=curve.at(k).x;
					mean_p.y+=curve.at(k).y;
				}else{

					if(main_angle<=1 && main_dist<=ak_dist){
						change=true;
						main_angle=ak_angle;
						new_curve.push_back(curve.at(k));
						mean_p.x+=curve.at(k).x;
						mean_p.y+=curve.at(k).y;

						main_dist=ak_dist;
					}else{
						ortho.x=0;first.x=0;secound.x=0;ak_vec.x=0;
						ortho.y=0;first.y=0;secound.y=0;ak_vec.y=0;
						main_angle=-1000;

						if(new_curve.size()>5){
							all_lines.push_back(new_curve);

							mean_p.x=mean_p.x/new_curve.size();
							mean_p.y=mean_p.y/new_curve.size();

							all_means.push_back(mean_p);
						}
					}

				}
				}
				else{//angle wechsel

				float akt_dist=sqrt(double(ak_vec.x*ak_vec.x)+double(ak_vec.y*ak_vec.y));


				if(main_angle<=ak_angle && ak_angle<=90 && akt_dist<=main_dist){
					main_angle=ak_angle;
					main_dist=ak_dist;

					new_curve.push_back(curve.at(k));
					mean_p.x+=curve.at(k).x;
					mean_p.y+=curve.at(k).y;
				}else{
					ortho.x=0;first.x=0;secound.x=0;ak_vec.x=0;
					ortho.y=0;first.y=0;secound.y=0;ak_vec.y=0;
					main_angle=-1000;

					if(new_curve.size()>5){
						all_lines.push_back(new_curve);
						
						mean_p.x=mean_p.x/new_curve.size();
						mean_p.y=mean_p.y/new_curve.size();

						all_means.push_back(mean_p);
						}
				}

				}






			}


			}//set first secound

		} //for


		


		
		
		//////// reverse

		ortho.x=0;first.x=0;secound.x=0;ak_vec.x=0;
		ortho.y=0;first.y=0;secound.y=0;ak_vec.y=0;
		

		main_angle=-1000;
		change=false;


		for(int k=curve.size()-1;k>-1;k--){

			if(first.x==0 && first.y==0){
				new_curve.clear();
				change=false;

				new_curve.push_back(curve.at(k));
				mean_p.x=curve.at(k).x;
				mean_p.y=curve.at(k).y;

				first.x=curve.at(k).x;
				first.y=curve.at(k).y;
			}else if(secound.x==0 && secound.y==0){

				new_curve.push_back(curve.at(k));
				mean_p.x+=curve.at(k).x;
				mean_p.y+=curve.at(k).y;

				secound.x=curve.at(k).x;
				secound.y=curve.at(k).y;
			}else{



			if(ortho.x==0 && ortho.y==0){
				ortho.x=secound.y-first.y;
				ortho.y=-(secound.x-first.x);
			}


			ak_vec.x=curve.at(k).x-first.x;
			ak_vec.y=curve.at(k).y-first.y;
			ak_angle=calc_angle(ak_vec, ortho);


			if(main_angle==-1000){
				

				if(ak_angle>=90){
					ortho.x=-ortho.x;
					ortho.y=-ortho.y;
					ak_angle=calc_angle(ak_vec, ortho);
				}

				if(ak_angle>=90){
					//not a curve
					ortho.x=0;first.x=0;secound.x=0;ak_vec.x=0;
					ortho.y=0;first.y=0;secound.y=0;ak_vec.y=0;
					main_angle=-1000;
				}else{
					main_angle=ak_angle;
					new_curve.push_back(curve.at(k));
					mean_p.x+=curve.at(k).x;
					mean_p.y+=curve.at(k).y;
				}


			}else{



				if(!change){
				if(main_angle>=ak_angle && ak_angle<=90){
					main_angle=ak_angle;
					new_curve.push_back(curve.at(k));
					mean_p.x+=curve.at(k).x;
					mean_p.y+=curve.at(k).y;
				}else{

					if(main_angle<=1){
						change=true;
						main_angle=ak_angle;
						new_curve.push_back(curve.at(k));
						mean_p.x+=curve.at(k).x;
						mean_p.y+=curve.at(k).y;
					}else{
						ortho.x=0;first.x=0;secound.x=0;ak_vec.x=0;
						ortho.y=0;first.y=0;secound.y=0;ak_vec.y=0;
						main_angle=-1000;

						if(new_curve.size()>5){
							all_lines.push_back(new_curve);

							mean_p.x=mean_p.x/new_curve.size();
							mean_p.y=mean_p.y/new_curve.size();

							all_means.push_back(mean_p);
						}
					}

				}
				}
				else{//angle wechsel
				if(main_angle<=ak_angle && ak_angle<=90){
					main_angle=ak_angle;

					new_curve.push_back(curve.at(k));
					mean_p.x+=curve.at(k).x;
					mean_p.y+=curve.at(k).y;
				}else{
					ortho.x=0;first.x=0;secound.x=0;ak_vec.x=0;
					ortho.y=0;first.y=0;secound.y=0;ak_vec.y=0;
					main_angle=-1000;

					if(new_curve.size()>5){
						all_lines.push_back(new_curve);
						
						mean_p.x=mean_p.x/new_curve.size();
						mean_p.y=mean_p.y/new_curve.size();

						all_means.push_back(mean_p);
						}
				}

				}






			}


			}//set first secound

		} //for



		////////


		

	}//line breakup






















	cv::RotatedRect selected_ellipse;


	for(int iii=0;iii<all_lines.size();iii++) {

		curve=all_lines.at(iii);
		mean_p=all_means.at(iii);

		int ergebniss=0;
		add_curve=true;

		cv::RotatedRect ellipse;

				for(int i=0;i<curve.size();i++) 
					if(  abs(mean_p.x-curve[i].x)<= mean_dist && abs(mean_p.y-curve[i].y) <= mean_dist)
						add_curve=false;





				//is ellipse fit possible
				if(add_curve){

					ellipse=cv::fitEllipse( cv::Mat(curve) );

					if(ellipse.center.x<0 || ellipse.center.y<0 || 
						ellipse.center.x>pic->cols || ellipse.center.y>pic->rows){

						add_curve=false;
					}


					
					if(ellipse.size.height > 3*ellipse.size.width ||
						ellipse.size.width > 3*ellipse.size.height){

						add_curve=false;
					}
					


					if(add_curve){ // pupil area
						if(ellipse.size.width*ellipse.size.height < pic->cols*pic->rows*0.005 ||
							ellipse.size.width*ellipse.size.height > pic->cols*pic->rows*0.2)//0.1
							add_curve=false;
					}

					
					if(add_curve){
						if(!is_good_ellipse_eval(&ellipse,pic,&ergebniss))
							add_curve=false;
					}
					
				}





				if(add_curve) {
					
					if(inner_color_range>=0){
						mean_inner_gray=0;




						mean_inner_gray=calc_inner_gray(pic,curve,ellipse);
						mean_inner_gray=mean_inner_gray*(1+abs(ellipse.size.height-ellipse.size.width));

						
						if(mean_inner_gray_last>mean_inner_gray){
							mean_inner_gray_last=mean_inner_gray;
							all_curves.clear();
							all_curves.push_back(curve);
						}else if(mean_inner_gray_last==mean_inner_gray){

							if(curve.size()>all_curves[0].size()){
								mean_inner_gray_last=mean_inner_gray;
								all_curves.clear();
								all_curves.push_back(curve);
								selected_ellipse=ellipse;
							}
						}
						



					}


				}


		}




		return all_curves;
}










static cv::RotatedRect find_best_edge(cv::Mat *pic,cv::Mat *edge, cv::Mat *magni, int start_x, int end_x, int start_y, int end_y, double mean_dist, int inner_color_range){

	cv::RotatedRect ellipse;
	ellipse.center.x=0;
	ellipse.center.y=0;
	ellipse.angle=0.0;
	ellipse.size.height=0.0;
	ellipse.size.width=0.0;

	std::vector<std::vector<cv::Point>> all_curves=get_curves(pic, edge, magni, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);





	if(all_curves.size()==1){
		ellipse=cv::fitEllipse( cv::Mat(all_curves[0]) );

		if(ellipse.center.x<0 || ellipse.center.y<0 || ellipse.center.x>pic->cols || ellipse.center.y>pic->rows){
			ellipse.center.x=0;
			ellipse.center.y=0;
			ellipse.angle=0.0;
			ellipse.size.height=0.0;
			ellipse.size.width=0.0;
		}

	}else{
		ellipse.center.x=0;
		ellipse.center.y=0;
		ellipse.angle=0.0;
		ellipse.size.height=0.0;
		ellipse.size.width=0.0;
	}

	return ellipse;

}

}