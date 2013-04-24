#include <stdio.h>

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
}eye_t;

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

void filter(const float *img, const int rows, const int cols, int * x_pos,int *y_pos,int *width)
// Algorithm inspired by:
// Robust real-time pupil tracking in highly off-axis images
// Lech Świrski Andreas Bulling Neil A. Dodgson
// Computer Laboratory, University of Cambridge, United Kingdom
// Eye Tracking Research & Applications 2012
{
    point_t img_size = {rows,cols};
    int min_h = 8;
    int max_h = 80;
    int h, i, j;
    float best_response = -10000;
    point_t best_pos ={0,0};
    int best_w = 0;
    int step = 4;

    for (h = min_h; h < max_h; h+=step)
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
                        best_w = eye.w;
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
    *width = best_w;
    }

