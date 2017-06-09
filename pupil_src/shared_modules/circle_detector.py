'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2
from methods import dist_pts_ellipse


def find_concetric_circles(gray_img,min_ring_count=3, visual_debug=False):
    # get threshold image used to get crisp-clean edges using blur to remove small features
    edges = cv2.adaptiveThreshold(cv2.blur(gray_img,(3,3)), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 11)
    _, contours, hierarchy = cv2.findContours(edges,
                                    mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_NONE,offset=(0,0)) #TC89_KCOS
    if visual_debug is not False:
        cv2.drawContours(visual_debug,contours,-1,(200,0,0))
    if contours is None or hierarchy is None:
        return []
    clusters = get_nested_clusters(contours,hierarchy[0],min_nested_count=min_ring_count)
    concentric_cirlce_clusters = []

    #speed up code by caching computed ellipses
    ellipses = {}

    # for each cluster fit ellipses and cull members that dont have good ellipse fit
    for cluster in clusters:
        if visual_debug is not False:
            cv2.drawContours(visual_debug, [contours[i] for i in cluster],-1, (0,0,255))
        candidate_ellipses = []
        for i in cluster:
            c = contours[i]
            if len(c)>5:
                if not i in ellipses:
                    e = cv2.fitEllipse(c)
                    fit = max(dist_pts_ellipse(e,c))
                    ellipses[i] = e,fit
                else:
                    e,fit = ellipses[i]
                a,b = e[1][0]/2.,e[1][1]/2.
                if fit<max(2,max(e[1])/20):
                    candidate_ellipses.append(e)
                    if visual_debug is not False:
                        cv2.ellipse(visual_debug, e, (0,255,0),1)

        if candidate_ellipses:
            cluster_center = np.mean(np.array([e[0] for e in candidate_ellipses]),axis=0)
            candidate_ellipses = [e for e in candidate_ellipses if np.linalg.norm(e[0]-cluster_center)<max(3,min(e[1])/20) ]
            if len(candidate_ellipses) >= min_ring_count:
                concentric_cirlce_clusters.append(candidate_ellipses)
                if visual_debug is not False:
                    cv2.ellipse(visual_debug, candidate_ellipses[-1], (0,255,255),4)

    #return clusters sorted by size of outmost cirlce biggest first.
    return sorted(concentric_cirlce_clusters,key=lambda e:-max(e[-1][1]))



def add_parents(child,graph,family):
    family.append(child)
    parent = graph[child,-1]
    if parent !=-1:
        family = add_parents(parent,graph,family)
    return family


def get_nested_clusters(contours,hierarchy,min_nested_count):
    clusters = {}
    # nesting of countours where many children are grouping in a sigle parent happens a lot (your screen with stuff in it. A page with text...)
    # we create a cluster for each of these children.
    # to reduce CPU load we onle keep the biggest cluster for clusters where the innermost parent is the same.
    for i in np.where(hierarchy[:,2]==-1)[0]: #contours with no children
        if len(contours[i])>=5: #we assume that valid markers innermost contour is longer than 5px
            cluster = add_parents(i,hierarchy,[])
            # is this cluster bigger that the current contender in the innermost parent group if if already exsists?
            if min_nested_count < len(cluster) > len(clusters.get(cluster[1],[])):
                clusters[cluster[1]] = cluster
    return clusters.values()


if __name__ == '__main__':
    def bench():
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,720)
        for x in range(100):
            sts,img = cap.read()
            # img = cv2.imread('/Users/mkassner/Desktop/manual_calibration_marker-01.png')
            gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            print(len(find_concetric_circles(gray,visual_debug=img)))
            cv2.imshow('img',img)
            cv2.waitKey(1)
            # return


    import cProfile,subprocess,os
    cProfile.runctx("bench()",{},locals(),"world.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
    print("created  time graph for  process. Please check out the png next to this file")
