'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
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


    #remove results which fully surround others, since we want the smalles ones
    cdef list bad = []
    cdef int x,y,w,x2,y2,w2
    cdef float response2

    bad = results[:]
    for r in reversed(results):
        x,y,w,response = r
        # filter for response
        if response < best_response*0.4 : #remove it anyway if it's bad
            results.remove(r)
            continue

        for g in bad:
            x2,y2,w2,response2 = g
            if x<x2 and y<y2 and x+w>x2+w2 and y+w>y2+w2: # g is fully included in r
                results.remove(r)
                break

    #calculate bounding box
    cdef int x_b = 0
    cdef int y_b = 0
    cdef int x2_b = 1
    cdef int y2_b = 1

    if len(results) > 0 :
        x_b , y_b  = results[0][:2]
        for v  in results:
            x,y,w,response = v
            x_b  = min(x, x_b)
            y_b  = min(y, y_b)
            if x2_b < x + w:
                x2_b = x+w
            if y2_b < y + w:
                y2_b = y+w

    return  (x_b , y_b, x2_b, y2_b) , results , bad
