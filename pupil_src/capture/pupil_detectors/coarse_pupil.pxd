'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

cimport cython
import math

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

    cdef list results = []

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
            results.append( (j ,i, h * 3 ,  response) )
            if len(results) > 30:
                results.pop(0)


    # filter for response
    results  = [ k  for k in results if k[3] > best_response*0.5 ]

    #group the nearest ones
    cdef dict group = {}
    cdef int x,y,w,xs,ys
    cdef float bucketSize = 20.0;
    for r in results:
        x,y,w,response = r
        xs = int(math.floor(x / bucketSize) * bucketSize )
        ys = int(math.floor(y / bucketSize) * bucketSize )

        if (xs,ys) in group:
            if group[(xs,ys)][2] > w and group[(xs,ys)][3] > response:
                group[(xs,ys)][2] = w
                group[(xs,ys)][3] = response
        else:
            group[(xs,ys)] = (x,y,w,response)


    #calculate bounding box
    cdef int x_b = 0
    cdef int y_b = 0
    cdef int x2_b = 1
    cdef int y2_b = 1

    if len(group) > 0 :
        x_b , y_b  = group.itervalues().next()[:2]
        for k ,v  in group.iteritems():
            x,y,w,response = v

            x_b  = min(x, x_b)
            y_b  = min(y, y_b)

            if x2_b < x + w:
                x2_b = x+w
            if y2_b < y + w:
                y2_b = y+w

    return  (x_b , y_b, x2_b, y2_b) , group
