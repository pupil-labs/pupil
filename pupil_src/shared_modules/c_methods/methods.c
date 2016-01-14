/*
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
*/

#include <stdio.h>


#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct {
   int    r;
   int    c;
} point_t;


typedef struct
{
    point_t s;
    point_t e;
    int a;
    float f;
} square_t;

typedef struct
{
    square_t outer;
    square_t inner;
    int w_half;
    int w;
    int h;
}eye_t;



typedef struct
{
    square_t outer;
    square_t middle;
    square_t inner;
    int w_i;
    int w_m;
    int w_o;
}ring_t;


// inline float area(const float *img,point_t size,point_t start,point_t end){
//     // use like in numpy including start excluding end
//     // 0-based indecies
//     // this is because the integral image has w+1 and h+1
//     return    img[end.r * size.c + end.c]
//             + img[start.r * size.c + start.c]
//             - img[start.r * size.c + end.c]
//             - img[end.r * size.c + start.c];
// }

inline float area(const float *img,point_t size,point_t start,point_t end,point_t offset){
    // use like in numpy including start excluding end
    // 0-based indecies
    // this is because the integral image has w+1 and h+1
    return    img[(offset.r + end.r  ) * size.c + offset.c + end.c]
            + img[(offset.r + start.r) * size.c + offset.c + start.c]
            - img[(offset.r + start.r) * size.c + offset.c + end.c]
            - img[(offset.r + end.r  ) * size.c + offset.c + start.c];
    }

inline eye_t make_eye(int h){
    int w = 3*h;
    eye_t eye;
    eye.h = h;
    eye.w = w;
    eye.outer.s = (point_t){0,0};
    eye.outer.e = (point_t){w,w};
    eye.inner.s = (point_t){h,h};
    eye.inner.e = (point_t){h+h,h+h};
    eye.inner.a = h*h;
    eye.outer.a = w*w;
    eye.outer.f =  1.0/eye.outer.a;
    eye.inner.f =  -1.0/eye.inner.a;
    eye.w_half = w/2;
    return eye;
}

inline ring_t make_ring(int w){
    ring_t ring;
    ring.w_i = w;
    ring.w_m = 3*w;
    ring.w_o = 9*w;
    ring.outer.s = (point_t){0,0};
    ring.outer.e = (point_t){9*w,9*w};
    ring.middle.s = (point_t){3*w,3*w};
    ring.middle.e = (point_t){6*w,6*w};
    ring.inner.s = (point_t){4*w,4*w};
    ring.inner.e = (point_t){5*w,5*w};
    ring.inner.a = w*w;
    ring.middle.a = 9*w*w;
    ring.outer.a = 81*w*w;
    ring.outer.f =  -.5/ring.outer.a;
    ring.middle.f =  1.0/ring.middle.a;
    ring.inner.f =  -.5/ring.inner.a;
    return ring;
}


// inline ring_t make_double_ring(int w){
//     ring_t ring;
//     ring.w_0 = w;
//     ring.w_1 = 3*w;
//     ring.w_2 = 9*w;
//     ring.w_3 = 15*w

//     ring.w_0.s = (point_t){4*w,4*w};
//     ring.w_0.e = (point_t){5*w,5*w};
//     ring.w_0.a = w*w;

//     ring.w_0.s = (point_t){4*w,4*w};
//     ring.w_0.e = (point_t){5*w,5*w};
//     ring.w_0.a = w*w;

//     ring.w_0.s = (point_t){4*w,4*w};
//     ring.w_0.e = (point_t){5*w,5*w};
//     ring.w_0.a = w*w;

//     ring.w_0.s = (point_t){4*w,4*w};
//     ring.w_0.e = (point_t){5*w,5*w};
//     ring.w_0.a = w*w;
//     ring.middle.a = 9*w*w;
//     ring.outer.a = 81*w*w;
//     ring.outer.f =  -.5/ring.outer.a;
//     ring.middle.f =  1.0/ring.middle.a;
//     ring.inner.f =  -.5/ring.inner.a;
//     return ring;
// }


void filter(const float *img, const int rows, const int cols, int * x_pos,int *y_pos,int *width, int min_w,int max_w,float *response)
// Algorithm based on:
// Robust real-time pupil tracking in highly off-axis images
// Lech Świrski Andreas Bulling Neil A. Dodgson
// Computer Laboratory, University of Cambridge, United Kingdom
// Eye Tracking Research & Applications 2012
{
    point_t img_size = {rows,cols};
    int min_h = min_w/3;
    int max_h = max_w/3;
    int h, i, j;
    float best_response = -10000;
    point_t best_pos ={0,0};
    int best_h = 0;
    int h_step = 4;
    int step = 5;

    for (h = min_h; h < max_h; h+=h_step)
        {
            eye_t eye = make_eye(h);
            // printf("inner factor%f outer.factor %f center %i \n",eye.inner.f,eye.outer.f,(int)eye.w_half );
            for (i=0; i<rows-eye.w; i +=step)
            {
                for (j=0; j<cols-eye.w; j+=step)
                {
                    // printf("|%2.0f",img[i * cols + j]);
                    point_t offset = {i,j};
                    float response =  eye.outer.f*area(img,img_size,eye.outer.s,eye.outer.e,offset)
                                     +eye.inner.f*area(img,img_size,eye.inner.s,eye.inner.e,offset);
                    // printf("| %5.2f ",response);
                    if(response > best_response){
                        // printf("!");
                        best_response = response;
                        best_pos = (point_t){i,j};
                        // printf("%i %i", (int)best_pos.r,(int)best_pos.c);
                        best_h = eye.h;
                    }
                }
                // printf("\n");
            }
        }



    // now we refine the search at pixel resolution This hole part can be commented out if needed
    point_t window_lower = {MAX(0,best_pos.r-step+1),MAX(0,best_pos.c-step+1)};
    point_t window_upper = {MIN(img_size.r,best_pos.r+step),MIN(img_size.c,best_pos.c+step)};
    for (h = best_h-h_step+1; h < best_h+h_step; h+=1)
        {
            eye_t eye = make_eye(h);
            // printf("inner factor%f outer.factor %f center %i \n",eye.inner.f,eye.outer.f,(int)eye.w_half );
            for (i=window_lower.r; i<MIN(window_upper.r,img_size.r-eye.w); i +=1)
            {
                for (j=window_lower.c; j<MIN(window_upper.c,img_size.c-eye.w); j +=1)
                {

                    // printf("|%2.0f",img[i * cols + j]);
                    point_t offset = {i,j};
                    float response =  eye.outer.f*area(img,img_size,eye.outer.s,eye.outer.e,offset)
                                     +eye.inner.f*area(img,img_size,eye.inner.s,eye.inner.e,offset);
                    // ikiuprintf("| %5.2f ",response);
                    if(response > best_response){
                        // printf("!");
                        best_response = response;
                        best_pos = (point_t){i,j};
                        // printf("%i %i", (int)best_pos.r,(int)best_pos.c);
                        best_h = eye.h;
                    }
                }
                // printf("\n");
            }
        }


    // point_t start = {0,0};
    // point_t end = {1,1};
    // printf("FULL IMG SUM %1.0f\n",img[(img_size.r-1) * img_size.c + (img_size.c-1)] );
    // printf("AREA:%f\n",area(img,img_size,start,end,(point_t){0,0}));
    *x_pos = (int)best_pos.r;
    *y_pos = (int)best_pos.c;
    *width = best_h*3;
    *response = best_response;
    }


void ring_filter(const float *img, const int rows, const int cols, int * x_pos,int *y_pos,int *width, float *response)
// Algorithm based on:
// Robust real-time pupil tracking in highly off-axis images
// Lech Świrski Andreas Bulling Neil A. Dodgson
// Computer Laboratory, University of Cambridge, United Kingdom
// Eye Tracking Research & Applications 2012
{
    point_t img_size = {rows,cols};
    int min_h = 6;
    int max_h = 50;
    int h, i, j;
    float best_response = -10000;
    point_t best_pos ={0,0};
    int best_h = 0;
    int h_step = 4;
    int step = 5;

    for (h = min_h; h < max_h; h+=h_step)
        {
            ring_t ring = make_ring(h);
            // printf("inner factor%f outer.factor %f center %i \n",ring.inner.f,ring.outer.f,(int)ring.w_o_half );
            for (i=0; i<rows-ring.w_o; i +=step)
            {
                for (j=0; j<cols-ring.w_o; j+=step)
                {
                    // printf("|%2.0f",img[i * cols + j]);
                    point_t offset = {i,j};
                    float response =  ring.outer.f*area(img,img_size,ring.outer.s,ring.outer.e,offset)
                                     +ring.middle.f*area(img,img_size,ring.middle.s,ring.middle.e,offset)
                                     +ring.inner.f*area(img,img_size,ring.inner.s,ring.inner.e,offset);
                    // printf("| %5.2f ",response);
                    if(response > best_response){
                        // printf("!");
                        best_response = response;
                        best_pos = (point_t){i,j};
                        // printf("%i %i", (int)best_pos.r,(int)best_pos.c);
                        best_h = ring.w_o;
                    }
                }
                // printf("\n");
            }
        }



    // now we refine the search at pixel resolution This hole part can be commented out if needed
    point_t window_lower = {MAX(0,best_pos.r-step+1),MAX(0,best_pos.c-step+1)};
    point_t window_upper = {MIN(img_size.r,best_pos.r+step),MIN(img_size.c,best_pos.c+step)};
    for (h = best_h-h_step+1; h < best_h+h_step; h+=1)
        {
            ring_t ring = make_ring(h);
            // printf("inner factor%f outer.factor %f center %i \n",ring.inner.f,ring.outer.f,(int)ring.w_o_half );
            for (i=window_lower.r; i<MIN(window_upper.r,img_size.r-ring.w_o); i +=1)
            {
                for (j=window_lower.c; j<MIN(window_upper.c,img_size.c-ring.w_o); j +=1)
                {

                    // printf("|%2.0f",img[i * cols + j]);
                    point_t offset = {i,j};
                     float response = ring.outer.f*area(img,img_size,ring.outer.s,ring.outer.e,offset)
                                     +ring.middle.f*area(img,img_size,ring.middle.s,ring.middle.e,offset)
                                     +ring.inner.f*area(img,img_size,ring.inner.s,ring.inner.e,offset);
                    // printf("| %5.2f ",response);
                    // ikiuprintf("| %5.2f ",response);
                    if(response > best_response){
                        // printf("!");
                        best_response = response;
                        best_pos = (point_t){i,j};
                        // printf("%i %i", (int)best_pos.r,(int)best_pos.c);
                        best_h = ring.w_o;
                    }
                }
                // printf("\n");
            }
        }


    // point_t start = {0,0};
    // point_t end = {1,1};
    // printf("FULL IMG SUM %1.0f\n",img[(img_size.r-1) * img_size.c + (img_size.c-1)] );
    // printf("AREA:%f\n",area(img,img_size,start,end,(point_t){0,0}));
    *x_pos = (int)best_pos.r;
    *y_pos = (int)best_pos.c;
    *width = best_h;
    *response = best_response;
    }

