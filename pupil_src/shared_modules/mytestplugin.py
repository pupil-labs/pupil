'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import logging
import cv2
import sys
import time
from plugin import Plugin

capture=None




logger = logging.getLogger(__name__)


class myExample_Plugin(Plugin):
    """docstring for Plugin

    plugin is a base class
    it has all interfaces that will be called
    instances of this class ususally get added to a plugins list
    this list will have its members called with all methods invoked.

    """
    def __init__(self,g_pool, atb_pos=(10,400)):
        Plugin.__init__(self)
        
        self._alive = True

        self.order = .6
        cascPath = 'haarcascade_frontalface_default.xml'
        # Create the haar cascade
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    #stream_server.handle_request()
        # between 0 and 1 this indicated where in the plugin excecution order you plugin lives:
        # <.5  are things that add/mofify information that will be used by other plugins and rely on untouched data.
        # You should not edit frame.img if you are here!
        # == 5 is the default.
        # >.5 are things that depend on other plugins work like display , saving and streaming

    @property
    def alive(self):
        """
        This field indicates of the instance should be detroyed
        Writing False to this will schedule the instance for deletion
        """
        if not self._alive:
            if hasattr(self,"cleanup"):
                    self.cleanup()
        return self._alive

    @alive.setter
    def alive(self, value):
        if isinstance(value,bool):
            self._alive = value

    def on_click(self,pos,button,action):
        """
        gets called when the user clicks in the window screen
        """
        pass

    def on_window_resize(self,window,w,h):
        '''
        gets called when user resizes window. 
        window is the glfw window handle of the resized window.
        '''
        pass
        
    def update(self,frame,recent_pupil_positions,events):
        """
        gets called once every frame
        if you plan to update the image data, note that this will affect all plugins axecuted after you.
        Use self.order to deal with this appropriately
        """
        image = frame.img
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow(gray)
        # Detect faces in the image
        faces = self.faceCascade.detectMultiScale(
                                     gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                                     )

        print "Found {0} faces!".format(len(faces))
        

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        pass



    def gl_display(self):
        """
        gets called once every frame when its time to draw onto the gl canvas.
        """
        pass


    def cleanup(self):
        """
        gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        pass

    def get_class_name(self):
        return self.__class__.__name__


    def __del__(self):
        self._alive = False





