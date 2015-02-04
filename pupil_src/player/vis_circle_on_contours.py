'''
(*)~----------------------------------------------------------------------------------

if gaze_point  is outside all contours:
    draw "outside"  
elif gaze_point  is inside contour_a:
    draw circle a
elif gaze_point  is inside contour_b:
    draw circle b
elif gaze_point  is (inside contour_a) and (inside contour_b):
    draw circle c
elif ...:
    draw circle ...

Author: Carlos Picanco, Universidade Federal do Para.
Hack from Pupil - eye tracking platform (v0.3.7.4):

plugin.py
vis_circle.py

Distributed under the terms of the CC BY-NC-SA License.
License details are in the file license.txt, distributed as part of this software.
    
----------------------------------------------------------------------------------~(*)
'''

from glfw import *
from plugin import Plugin
from ctypes import c_int,c_float,c_bool
from gl_utils import adjust_gl_view


from methods import denormalize
from player_methods import transparent_circle
from vcc_methods import get_canditate_ellipses, ellipses_from_findContours, get_cluster_hierarchy, ellipse_to_contour
from vcc_methods import PolygonTestEx, PolygonTestRC, get_codes
from vcc_methods import find_edges, draw_contours


import numpy as np
# import OpenGL.GL as gl
import cv2
import logging
import platform 
logger = logging.getLogger(__name__)

# pt_codes references
_POINT = 0
_CODE = 1

# channel constants
_CH_B = 0  
_CH_G = 1
_CH_R = 2
_CH_0 = 3

class Circle_on_Contours(Plugin):
    def __init__(
        self,
        g_pool = None,
        radius = 20,
        color3 = (0.1,.2,.1,.5),
        thickness = 1,
        full = False,
        epsilon = 0.007,
        detection_method_idx = 1,
        blackbkgnd = False):
        super(Circle_on_Contours, self).__init__()
        # pupil standards
        self.g_pool = g_pool 
        self.order = .5

        # circle
        self.radius = c_int(int(radius))
        self.color3 = (c_float * 4)(*color3)
        self.thickness = c_int(int(thickness))
        self.full = c_bool(bool(full))

        # world window
        self.blackbkgnd = c_bool(bool(blackbkgnd))

        # plugin window
        self.window_should_open = False
        self.window_should_close = False
        self._window = None
        self.fullscreen = c_bool(0)
        self.monitor_idx = c_int(0)
        self.monitor_handles = glfwGetMonitors()
        self.monitor_names = [glfwGetMonitorName(m) for m in self.monitor_handles]

        # detector
        self.candraw = False
        self.detection_color = (c_float*4)(*color3)
        self.detection_method_idx = c_int(2)
        self.expected_contours = c_int(2)
        self.detection_method_names = ['Method 1', 'Method 2', 'Method 3']
        self.threshold = c_int(255)
        self.ellipse_size = c_float(2.000)
        self.channelidx = c_int(6)
        self.channel_names = ['B', 'G', 'R', 'GR', 'BR', 'BG', 'BGR']
        self.epsilon = c_float(epsilon)

        #Method 2 uses screen_marker_calibration/circle_detector functions
        self.candidate_ellipses = []
        self.current_path = 'none'
        self.timestamps = []
        self.current_frame = []

        # hardcoded colors, but they should be assigned at runtime
        self.colors = [  (0, 0, 255, 150),      # red
                    (255, 0, 0, 150),       # blue
                    (0, 255, 0, 150),       # green
                    (255, 255, 0, 150),     # blue marine
                    (0, 255, 255, 150)      # yellow
                    #(230, 50, 230, 150),     # purple
                    #(0, 0, 255, 200),      # red
                    #(255, 255, 255, 200),  # white
                    #(0, 0, 0, 200)         # black
                ]
        self.codes = list(get_codes('-+', self.expected_contours.value))
        # find a way to choose colors for larger numbers   ]

        self.ColorDictionary = dict(zip(self.codes, self.colors))
        #self.ColorDictionary['+1'] = (230, 50, 230, 150)
        #self.ColorDictionary['-1'] = (0, 0, 0, 255) 

        logger.info('Circle_on_Contours plugin initialization on ' + platform.system())
        #primary_monitor = glfwGetPrimaryMonitor()
    def get_variables(self):
        # contours, cntcount, cntellps, elpcount, pupilsxy, pxycount, pxytests, elp_alfa
        pass

    def update(self,frame,recent_pupil_positions,events):
        #logger.debug(len(frame.img))
        self.current_frame = frame.index

        ############################################################################################
        #"""METHOD 1"""#############################################################################    
        ############################################################################################
        if self.detection_method_idx.value == 0:
            '''

            self.edgesidx.value['B', 'G', 'R', 'GR', 'BR', 'BG', 'BGR']
            
            edges[_CH_B, _CH_G, _CH_R, _CH_0]

            img.shape[height, width, channels]


            '''
            img = frame.img
            height = img.shape[0] 
            width = img.shape[1]

            # cv2.THRESH_BINARY | cv2.THRESH_OTSU
            # cv2.THRESH_BINARY_INV
            # cv2.THRESH_TRUNC
            # cv2.THRESH_TOZERO
            # cv2.THRESH_TOZERO_INV    
            edges = find_edges(img, self.threshold.value,cv2.THRESH_TOZERO)
            #edges = find_edges2(img, self.threshold.value)
            edges.append(np.zeros((height, width, 1), np.uint8))

            # logger.debug(str(cv2.cv.GetImage(cv2.cv.fromarray(frame.img)).depth)) # 8
            #logger.debug(str(frame.img.))

            if self.channelidx.value == 6:
                
                edges_edt = cv2.max(edges[_CH_B], edges[_CH_G])
                edges_edt = cv2.max(edges_edt, edges[_CH_R])
                
                frame.img = cv2.merge([edges_edt, edges_edt, edges_edt])
                #, edges[_CH_R]]
                #edgs = cv2.max(edges[_CH_B], edges[_CH_G])

            #    in_put = [edges[_CH_B], edges[_CH_G], edges[_CH_R]]
            #    out_put = [edges[_CH_B], edges[_CH_G], edges[_CH_R]]

                #from_to = [ _CH_R,_CH_B,  _CH_B,_CH_G,  _CH_G,_CH_B ]  
            #    from_to = [2,0, 0,1, 1,0]
                #edges = cv2.merge(in_put)
            #    cv2.mixChannels(in_put, out_put, from_to)
            #    edges = cv2.merge(out_put)
            #    frame.img = edges
                #self.frame = edges
            #    edgs = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)

            elif self.channelidx.value == 5:
                edges_edt = cv2.max(edges[_CH_B], edges[_CH_G]) 
                frame.img = cv2.merge([edges_edt, edges_edt, edges[_CH_0]])

            elif self.channelidx.value == 4:
                edges_edt = cv2.max(edges[_CH_B], edges[_CH_R])
                frame.img = cv2.merge([edges_edt, edges[_CH_0], edges_edt])

            elif self.channelidx.value == 3:
                edges_edt = cv2.max(edges[_CH_G], edges[_CH_R])
                frame.img = cv2.merge([edges[_CH_0],edges_edt, edges_edt])

            elif self.channelidx.value == _CH_R:
                edges_edt = edges[_CH_R]
                frame.img = cv2.merge([ edges[_CH_0], edges[_CH_0], edges_edt ])

            elif self.channelidx.value == _CH_G:
                edges_edt = edges[_CH_G]
                frame.img = cv2.merge([ edges[_CH_0], edges_edt, edges[_CH_0] ])  

            elif self.channelidx.value == _CH_B:
                edges_edt = edges[_CH_B]
                frame.img = cv2.merge([ edges_edt, edges[_CH_0], edges[_CH_0] ]) 

            # error: (-210) [Start]FindContours support only 8uC1 and 32sC1 images in function cvStartFindContours
            contours,hierarchy = cv2.findContours(edges_edt, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE,offset=(0,0))

            #self.contours = contours

            color1 = map(lambda x: int(x * 255),self.detection_color)
            color1 = color1[:3][::-1] + color1[-1:]
            
            contours = draw_contours(frame.img, contours, hierarchy, self.epsilon.value, color1)

            #logger.info(str(len(contours)) + ' contours found.')
            
            color1 = map(lambda x: int(x * 255),self.detection_color)
            color1 = color1[:3][::-1] + color1[-1:]
            color2 = (255, 0, 0, 255)
            
            if self.full.value:
                thickness= -1
            else:
                thickness = self.thickness.value

            radius = self.radius.value
            pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y = True) for pt in recent_pupil_positions if pt['norm_gaze'] is not None]
            #logger.info(str(len(pts)) + ' gaze points found.')
            # logger.debug(str(pt))
            PolygonTests = []
            for pt in pts:
                for contour in contours:
                    Inside = cv2.pointPolygonTest(contour, pt, False)
                    if Inside > -1:
                        PolygonTests.append(True)
                    else:
                        PolygonTests.append(False)
                if True in PolygonTests:
                    transparent_circle(frame.img, pt, radius = 10, color = color1, thickness = thickness)
                else:
                    transparent_circle(frame.img, pt, radius = radius, color = color2, thickness = thickness)
        
        ############################################################################################
        #"""METHOD 2"""#############################################################################    
        ############################################################################################       
        if self.detection_method_idx.value == 1:
            # Get image from frame       
            img = frame.img
            #height = img.shape[0] 
            #width = img.shape[1]

            # Ctypes for Gui compatibility
            show_edges = c_bool(0)
            dist_threshold = c_int(20)
            area_threshold = c_int(100)

            # color3    
            color3 = map(lambda x: int(x * 255),self.detection_color)
            color3 = color3[:3][::-1] + color3[-1:]
            
            # cv2.THRESH_BINARY
            # cv2.THRESH_BINARY_INV
            # cv2.THRESH_TRUNC
            candidate_ellipses = []
            remainders = []

            candidate_ellipses, remainders = get_canditate_ellipses(img,
                                                            img_threshold=self.threshold.value,
                                                            cv2_thresh_mode=cv2.THRESH_BINARY,    
                                                            area_threshold=area_threshold.value,
                                                            dist_threshold=dist_threshold.value,
                                                            min_ring_count=3,
                                                            visual_debug=show_edges.value)
            contours_target = []

            if len(candidate_ellipses) > 0:
                ellipse = candidate_ellipses[-1]
                # norm_pos = normalize(orig_pos,(img.shape[1],img.shape[0]),flip_y=True)
                center = (int(round(ellipse[0][0])),int(round(ellipse[0][1]))) 
                axes = (int(round(ellipse[1][0]/self.ellipse_size.value)),int(round(ellipse[1][1]/self.ellipse_size.value)))
                angle = int(round(ellipse[2]))

                contour = cv2.ellipse2Poly(center, axes, angle,
                                                               arcStart=0,
                                                               arcEnd=360,
                                                               delta=1) # precision angle
                contours_target.append(contour)
                cv2.drawContours(frame.img, contours_target, -1, color3, thickness=1,lineType=cv2.CV_AA)

            contours_remainders = []

            if len(remainders) > 0:
                ellipse = remainders[-1]
                # norm_pos = normalize(orig_pos,(img.shape[1],img.shape[0]),flip_y=True)
                center = (int(round(ellipse[0][0])),int(round(ellipse[0][1]))) 
                axes = (int(round(ellipse[1][0]/self.ellipse_size.value)),int(round(ellipse[1][1]/self.ellipse_size.value)))
                angle = int(round(ellipse[2]))
                # logger.debug(str(len(axis)))

                contour = cv2.ellipse2Poly(center, axes, angle,
                                                               arcStart=0,
                                                               arcEnd=360,
                                                               delta=1)    # precision angle
                
                contours_remainders.append(contour)
                cv2.drawContours(frame.img, contours_remainders, -1, (255, 0, 0), thickness=1,lineType=cv2.CV_AA)
            
            pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y = True) for pt in recent_pupil_positions if pt['norm_gaze'] is not None]
            pt_codes = []
            contours_counter = 0
            for pt in pts:
                contours_counter, counter_code = PolygonTestEx(contours_target, pt)
                contours_counter, counter_code = PolygonTestEx(contours_remainders, pt, contours_counter, counter_code)
                pt_codes.append((pt, counter_code))

            # transparent circle parameters
            if self.full.value:
                thickness= -1
            else:
                thickness = self.thickness.value
            radius = self.radius.value

            # need to draw contours in the same order 
            if contours_counter > 0:
                for pt_code in pt_codes:
                    try:
                        color = self.ColorDictionary[pt_code[_CODE]]
                    except KeyError, e:
                        #print e
                        color = (0, 0, 0, 255)

                    transparent_circle(
                                frame.img,
                                pt_code[_POINT],
                                radius = int(radius/2),
                                color = color,
                                thickness = thickness    )
            else:
                for pt in pts:
                    transparent_circle(
                        frame.img,
                        pt,
                        radius = radius,
                        color = self.colors[-1],
                        thickness = thickness    )
                    cv2.putText(frame.img, '?', (int(pt[0] -10),int(pt[1]) +10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, lineType = cv2.CV_AA )
       
        ############################################################################################
        #"""METHOD 3"""#############################################################################    
        ############################################################################################
        if self.detection_method_idx.value == 2:
            # get image from frame       
            img = frame.img

            # set color3    
            color3 = map(lambda x: int(x * 255),self.detection_color)
            color3 = color3[:3][::-1] + color3[-1:]

            # cv2.THRESH_BINARY
            # cv2.THRESH_BINARY_INV
            # cv2.THRESH_TRUNC

            # find raw ellipses from cv2.findContours
            show_edges = c_bool(0)

            # the less the difference between ellipse area and source contour area are,
            # the better a fit between ellipse and source contour will be
            # delta_area_threshold gives the maximum allowed difference
            delta_area_threshold = c_int(20)
            ellipses = []
            ellipses = ellipses_from_findContours(img,
                                    cv2_thresh_mode=cv2.THRESH_BINARY,    
                                    delta_area_threshold=delta_area_threshold.value,
                                    visual_debug=show_edges.value)

            # we need denormalized points for point polygon tests    
            pts = [denormalize(pt['norm_gaze'], img.shape[:-1][::-1], flip_y = True) for pt in recent_pupil_positions if pt['norm_gaze'] is not None]
               
            if ellipses:
                # get area of all ellipses
                ellipses_temp = [e[1][0]/2. * e[1][1]/2. * np.pi for e in ellipses]
                ellipses_temp.sort()

                # take the highest area as reference
                area_threshold = ellipses_temp[-1]
            
                # filtering by proportional area
                ellipses_temp = []
                for e in ellipses:
                    a,b = e[1][0] / 2., e[1][1] / 2.
                    ellipse_area = np.pi * a * b
                    if (ellipse_area/area_threshold) < .10:
                        pass  
                    else:
                        ellipses_temp.append(e)

                # cluster_hierarchy is ordenated by appearence order, from top left screen
                # it is a list of clustered ellipses
                dist_threshold = c_int(20)
                cluster_hierarchy = []
                cluster_hierarchy = get_cluster_hierarchy(
                                        ellipses=ellipses_temp,
                                        dist_threshold=dist_threshold.value)
                # total_stm is expected to be the number of stimuli on screen
                # total_stm = len(cluster_hierarchy)

                # we need contours for point polygon tests, not ellipses
                stm_contours = []

                # cluster_set is the ellipse set associated with each stimulus on screen
                alfa = self.ellipse_size.value

                temp = list(cluster_hierarchy)
                for cluster_set in temp:
                    print len(cluster_set)
                    if len(cluster_set) > 2:
                        cluster_hierarchy.append(cluster_hierarchy.pop(cluster_hierarchy.index(cluster_set)))

                for cluster_set in cluster_hierarchy:
                    if len(cluster_set) > 0:
                        if True:
                            for ellipse in cluster_set:
                                center = ( int(round( ellipse[0][0] )), int( round( ellipse[0][1] ))) 
                                axes = ( int( round( ellipse[1][0]/alfa )), int( round( ellipse[1][1]/alfa )))
                                angle = int( round(ellipse[2] ))
                                cv2.ellipse(img, center, axes, angle, startAngle=0, endAngle=359, color=color3, thickness=1, lineType=8, shift= 0)

                        # use only the biggest (last) ellipse for reference
                        stm_contours.append(ellipse_to_contour(cluster_set[-1], alfa))

                #print stm_contours
                # pt_codes is a list tuples:
                # tuple((denormalized point as a float x, y coordenate), 'string code given by the PointPolygonTextEx function')
                # ex.: tuple([x, y], '+1-2')
                contour_count = 0
                pt_codes = []
                for pt in pts:
                    contour_count = 0
                    counter_code = ''
                    for contour in stm_contours:
                        contour_count, counter_code = PolygonTestRC(contour, pt, contour_count, counter_code)
                    # a single code for a single point
                    pt_codes.append((pt, counter_code))
                print pt_codes

            else:
                print 'else'
                contour_count = 0
               
            # transparent circle parameters
            radius = self.radius.value
            if self.full.value:
                thickness= -1
            else:
                thickness = self.thickness.value 

            # each code specifies the color of each point
            # in accordance with the self.ColorDictionary
            if contour_count > 0:
                for x in xrange(len(pt_codes)):
                    try:
                        color = self.ColorDictionary[pt_codes[x][_CODE]]
                    except KeyError, e:
                        #print e
                        color = (0, 0, 0, 255)

                    transparent_circle(
                                img,
                                pt_codes[x][_POINT],
                                radius = int(radius/2),
                                color = color,
                                thickness = thickness    )
            else:
                for pt in pts:
                    transparent_circle(
                        frame.img,
                        pt,
                        radius = radius,
                        color = self.colors[-1],
                        thickness = thickness    )
                    cv2.putText(img, '?', (int(pt[0] -10),int(pt[1]) +10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, lineType = cv2.CV_AA )

        if self.window_should_close:
            pass # self.close_window()

        if self.window_should_open:
            pass # self.open_window()

    def init_gui(self):
        import atb

        atb_pos = 630, 10
        # creating an AntTweakBar.
        atb_label = "Circle on Contours"
        self._bar = atb.Bar(
            name = self.__class__.__name__,
            label = atb_label,
            help = "ref detection parameters",
            color = (50, 50, 50),
            alpha = 100,
            text = 'light',
            position = atb_pos,
            refresh = .3,
            size = (300, 300))

        # circle parameters
        self._bar.add_var('circle radius',self.radius)
        self._bar.add_var('circle thickness',self.thickness, min = 0)
        self._bar.add_var('circle filled',self.full)
        self._bar.add_separator('sep0')

        # detector
        self._bar.add_var('target color',self.detection_color)
        self._bar.add_var('black background', self.blackbkgnd)
        self._bar.add_var('threshold', self.threshold, min = 0, max = 255, step = 1)
        self._bar.add_var('Epsilon',self.epsilon, min = 0, max = 5, step = 0.0001)
        detection_method = atb.enum("Method",dict(((key,val) for val,key in enumerate(self.detection_method_names)))) 
        self._bar.add_var("Detection Method",self.detection_method_idx, vtype = detection_method)
        edges_channel = atb.enum("Channel",dict(((key,val) for val,key in enumerate(self.channel_names)))) 
        self._bar.add_var("Channels",self.channelidx, vtype = edges_channel)

        self._bar.add_separator('sep1')
        self._bar.add_var('Expected Contours', self.expected_contours, min = 2, max = 32, step= 1)
        self._bar.add_var('Ellipse', self.ellipse_size, min = 0, max = 4, step = 0.001)
        
        self._bar.add_separator('sep2')

        # window parameters
        self._bar.add_button('remove',self.unset_alive)
        self._bar.add_button('debug button', self.debug_button)

    def unset_alive(self):
        self.alive = False

    def debug_button(self):
        print self.debug

    def on_plugin_window_key(self,window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    self.on_close()

    def on_close(self,window = None):
        self.window_should_close = True
       
    def get_init_dict(self):
        return {
            'radius':self.radius.value,
            'color3':self.detection_color[:],
            'thickness':self.thickness.value,
            'full':self.full.value,
            'epsilon': self.epsilon.value,
            'detection_method_idx':self.detection_method_idx.value,
            'blackbkgnd': self.blackbkgnd.value}

    def clone(self):
        return Circle_on_Contours(**self.get_init_dict())

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happends either volunatily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self._window:
            pass # self.close_window()
        self._bar.destroy()