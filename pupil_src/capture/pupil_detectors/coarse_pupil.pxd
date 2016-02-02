'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

cimport cython

cdef struct point_t :
   int    r
   int    c

cdef struct square_t:
    point_t s
    point_t e
    int a
    float f

cdef struct eye_t:
    square_t outer
    square_t inner
    int w_half
    int w
    int h


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int area(int[:,::1] img,point_t size,point_t start,point_t end,point_t offset):
  with cython.boundscheck(False):  # cython bug! ignores @cython.wraparound(False)
      return    img[offset.r + end.r ,offset.c + end.c]\
            + img[offset.r + start.r, offset.c + start.c]\
            - img[offset.r + start.r, offset.c + end.c]\
            - img[offset.r + end.r,offset.c + start.c]

@cython.cdivision(True)
cdef inline eye_t make_eye(int h) nogil:
    cdef int w = 3*h
    cdef eye_t eye
    eye.h = h
    eye.w = w
    eye.outer.s.r = 0
    eye.outer.s.c = 0
    eye.outer.e.r = w
    eye.outer.e.c = w
    eye.inner.s.r = h
    eye.inner.s.c = h
    eye.inner.e.r = h+h
    eye.inner.e.c = h+h
    eye.inner.a = h*h
    eye.outer.a = w*w
    eye.outer.f =  1.0/eye.outer.a
    eye.inner.f =  -1.0/eye.inner.a
    eye.w_half = w/2
    return eye

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline center_surround(int[:,::1] img, int min_w,int max_w):
    cdef point_t img_size
    img_size.r =  img.shape[0]
    img_size.c =  img.shape[1]
    cdef int min_h = min_w/3
    cdef int max_h = max_w/3
    cdef int h=0, i=0, j=0
    cdef float best_response = -10000
    cdef point_t best_pos
    cdef int best_h = 0
    cdef int h_step = 4
    cdef int step = 5
    cdef eye_t eye
    cdef point_t offset
    cdef int x_pos,y_pos,width
    cdef float response = 0
    cdef float a,c,
    cdef point_t b,d,e,f


    #for h in prange(min_h,max_h,h_step):
    for h from min_h <= h < max_h by h_step:
      eye = make_eye(h)
      #for i in range(0,img_size.r-eye.w,step): #step is slow
      for i from 0 <= i < img_size.r-eye.w by step:
        #for j in range(0,img_size.c-eye.w,step): #step is slow
        for j from 0 <= j < img_size.c-eye.w by step:

          offset.r = i
          offset.c = j

          response = eye.outer.f*area(img,img_size,eye.outer.s,eye.outer.e,offset) + eye.inner.f*area(img,img_size,eye.inner.s,eye.inner.e,offset)
          if(response  > best_response):
            best_response = response
            best_pos.r = i
            best_pos.c = j
            best_h = h


    cdef point_t window_lower
    window_lower.r = max(0,best_pos.r-step+1)
    window_lower.c = max(0,best_pos.c-step+1)


    cdef point_t window_upper
    window_upper.r = min(img_size.r,best_pos.r+step)
    window_upper.c = min(img_size.c,best_pos.c+step)

    for h in range(max(3,best_h-h_step+1),best_h+h_step):
        eye = make_eye(h)
        for i in range(window_lower.r,min(window_upper.r,img_size.r-eye.w)) :
            for j in range(window_lower.c,min(window_upper.c,img_size.c-eye.w)):
                offset.r = i
                offset.c = j
                response = eye.outer.f*area(img,img_size,eye.outer.s,eye.outer.e,offset) + eye.inner.f*area(img,img_size,eye.inner.s,eye.inner.e,offset)
                if(response > best_response):
                    best_response = response
                    best_pos.r = i
                    best_pos.c = j
                    best_h = h

    x_pos = best_pos.c
    y_pos = best_pos.r
    width = best_h*3
    return x_pos,y_pos,width,best_response
