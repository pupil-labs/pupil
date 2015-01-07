'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import logging
logger = logging.getLogger(__name__)

class Plugin(object):
    """docstring for Plugin
    plugin is a base class
    it has all interfaces that will be called
    instances of this class ususally get added to a plugins list
    this list will have its members called with all methods invoked.

    """
    # if you have a plugin that can exsist multiple times make this false in your derived class
    is_unique = True

    def __init__(self,g_pool):
        self._alive = True
        self.g_pool = g_pool
        self.order = .5
        # between 0 and 1 this indicated where in the plugin excecution order you plugin lives:
        # <.5  are things that add/mofify information that will be used by other plugins and rely on untouched data.
        # You should not edit frame if you are here!
        # == .5 is the default.
        # >.5 are things that depend on other plugins work like display , saving and streaming

    def init_gui(self):
        '''
        if the app allows a gui, you may initalize your part of it here.
        '''
        pass


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
        if you have an gui or glfw window destroy it here.
        """
        pass



    def on_click(self,pos,button,action):
        """
        gets called when the user clicks in the window screen
        """
        pass

    @property
    def class_name(self):
        '''
        name of this instance's class
        '''
        return self.__class__.__name__

    @property
    def base_class(self):
        '''
        base class of this instance's class
        '''
        return self.__class__.__bases__[0]

    @property
    def base_class_name(self):
        '''
        base class name of this instance's class
        '''
        return self.base_class.__name__

    @property
    def pretty_class_name(self):
        return self.class_name.replace('_',' ')


    ### if you want a session persistent plugin implement this function:
    # def get_init_dict(self):
    #     d = {}
    #     # add all aguments of your plugin init fn with paramter names as name field
    #     # do not include g_pool here.
    #     return d

    def __del__(self):
        self._alive = False



# Derived base classes:
# If you inherit from these your plugin property base_class will point to them
# This is good because we can categorize plugins.
class Calibration_Plugin(Plugin):
    '''base class for all calibration routines'''
    pass

class Gaze_Mapping_Plugin(Plugin):
    '''base class for all calibration routines'''
    pass
