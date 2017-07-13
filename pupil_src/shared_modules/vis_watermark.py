'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from player_methods import transparent_image_overlay
from plugin import Visualizer_Plugin_Base
import numpy as np
import cv2
from glob import glob
import os
from pyglui import ui

from glfw import glfwGetCursorPos,glfwGetWindowSize,glfwGetCurrentContext
from methods import normalize,denormalize
import logging
logger = logging.getLogger(__name__)


class Vis_Watermark(Visualizer_Plugin_Base):
    uniqueness = "not_unique"

    def __init__(self, g_pool,selected_watermark_path = None,pos = (20,20)):
        super().__init__(g_pool)
        self.order = .9
        self.menu = None

        available_files = glob(os.path.join(self.g_pool.user_dir,'*png')) #we only look for png's
        self.available_files = [f for f in available_files if cv2.imread(f,-1).shape[2]==4] #we only look for rgba images
        logger.debug('Found {} watermark files: {}'.format(len(self.available_files), self.available_files))

        self.watermark = None
        self.watermark_path = None
        self.alpha_mask = None

        if selected_watermark_path in self.available_files:
            self.load_watermark(selected_watermark_path)
        elif self.available_files:
            self.load_watermark(self.available_files[0])
        else:
            logger.warning("No .png files found. Make sure they are in RGBA format.")

        self.pos = list(pos) #if we make the default arg a list the instance will edit the default vals and a new call of the class constructor creates an ainstace with modified default values.
        self.move_watermark = False
        self.drag_offset = None

    def load_watermark(self,path):
        img = cv2.imread(path,-1)
        if img is None:
            logger.error("Could not load watermark img file.")
        else:
            self.watermark = img[:,:,:3]
            self.alpha_mask = img[:,:,3]/255.0
            self.alpha_mask = np.dstack((self.alpha_mask,self.alpha_mask,self.alpha_mask))
            self.watermark_path = path

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        if self.drag_offset is not None:
            pos = glfwGetCursorPos(glfwGetCurrentContext())
            pos = normalize(pos,glfwGetWindowSize(glfwGetCurrentContext()))
            pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
            self.pos[0] = pos[0]+self.drag_offset[0]
            self.pos[1] = pos[1]+self.drag_offset[1]

        if self.watermark is not None:
            #keep in image bounds, do this even when not dragging because the image sizes could change.
            self.pos[1] = max(0,min(frame.img.shape[0]-self.watermark.shape[0],max(self.pos[1],0)))
            self.pos[0] = max(0,min(frame.img.shape[1]-self.watermark.shape[1],max(self.pos[0],0)))
            pos = int(self.pos[0]),int(self.pos[1])
            img  = frame.img
            roi = slice(pos[1],pos[1]+self.watermark.shape[0]),slice(pos[0],pos[0]+self.watermark.shape[1])
            w_roi = slice(0,img.shape[0]-pos[1]),slice(0,img.shape[1]-pos[0])
            img[roi] = self.watermark[w_roi]*self.alpha_mask[w_roi] + img[roi]*(1-self.alpha_mask[w_roi])

    def on_click(self,pos,button,action):
        if self.move_watermark and action == 1:
            if self.pos[0] < pos[0] < self.pos[0]+ self.watermark.shape[0] and self.pos[1] < pos[1] < self.pos[1]+ self.watermark.shape[1]:
                self.drag_offset = self.pos[0]-pos[0],self.pos[1]-pos[1]
        else:
            self.drag_offset = None

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Watermark')
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close',self.unset_alive))
        if self.watermark is None:
            self.menu.append(ui.Info_Text("Please save a .png file in the users settings dir: '{}' in RGBA format. Once this plugin is closed and re-loaded the png will be used as a watermark.".format(self.g_pool.user_dir)))
        else:
            if len(self.available_files) > 1:
                self.menu.append(ui.Selector("watermark_path",self,label='file',selection= self.available_files,labels= [os.path.basename(p) for p in self.available_files], setter=self.load_watermark))
            self.menu.append(ui.Switch('move_watermark',self))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def gl_display(self):
        if self.move_watermark:
            pass

    def get_init_dict(self):
        return {'selected_watermark_path':self.watermark_path,'pos':tuple(self.pos)}

    def cleanup(self):
        """called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
