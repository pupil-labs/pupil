'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2

def get_candidate_ellipses(gray_img,area_threshold,dist_threshold,min_ring_count, visual_debug):

    # get threshold image used to get crisp-clean edges
    edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, -3)
    # cv2.flip(edges,1 ,dst = edges,)
    # display the image for debugging purpuses
    # img[:] = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    # from edges to contours to ellipses CV_RETR_CCsOMP ls fr hole
    _ ,contours, hierarchy = cv2.findContours(edges,
                                    mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS

    # remove extra encapsulation
    if contours is None or hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    # turn outmost list into array
    contours =  np.array(contours)
    # keep only contours                        with parents     and      children
    contained_contours = contours[np.logical_and(hierarchy[:,3]>=0, hierarchy[:,2]>=0)]
    # turn on to debug contours
    if visual_debug:
        cv2.drawContours(gray_img, contained_contours,-1, (0,0,255))

    # need at least 5 points to fit ellipse
    contained_contours =  [c for c in contained_contours if len(c) >= 5]

    ellipses = [cv2.fitEllipse(c) for c in contained_contours]
    candidate_ellipses = []
    # filter for ellipses that have similar area as the source contour
    for e,c in zip(ellipses,contained_contours):
        a,b = e[1][0]/2.,e[1][1]/2.
        if abs(cv2.contourArea(c)-np.pi*a*b) <area_threshold:
            candidate_ellipses.append(e)




    candidate_ellipses = get_cluster(candidate_ellipses,dist_threshold = dist_threshold,min_ring_count=min_ring_count)

    return candidate_ellipses


def man_dist(e,other):
    return abs(e[0][0]-other[0][0])+abs(e[0][1]-other[0][1])

def get_cluster(ellipses,dist_threshold,min_ring_count):
    for e in ellipses:
        close_ones = []
        for other in ellipses:
            # distance to other ellipse is smaller than min dist threshold and minor of both ellipses
            if man_dist(e,other)<dist_threshold and man_dist(e,other) < min(min(*e[1]),min(other[1])) :
                close_ones.append(other)
        if len(close_ones)>=min_ring_count:
            # sort by major axis to return smallest ellipse first
            close_ones.sort(key=lambda e: max(e[1]))
            return close_ones
    return []


