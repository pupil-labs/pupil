'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from raw import *
#logging
import logging
logger = logging.getLogger(__name__)

class Control(dict):
    """
    docstring
    """
    def __init__(self,c):
        for key,val in c.items():
            self[key] = val

        self.name = self['name']
        self.atb_name = self['name']
        self.order = self['order']
        self.value = self['value']
        self.default = self['default']
        self.type  = self['type']
        if self.type == 'menu':
            self.menu = self['menu']
            self.min = None
            self.max = None
            self.step = None
        elif self.type == 'bool':
            self.min = 0
            self.max = 1
            self.step = 1
            self.menu=None
        else:
            self.menu=None
            self.step = self['step']
            self.min = self['min']
            self.max = self['max']

        if 'flags' in self:
            self.flags = self['flags']
        else:
            self.flags = "active"


    def get_val(self):
        return self['value']

    def set_val(self,val):
        set(self['src'],self['name'],val)
        self['value'] = val


class Controls(dict):
    """docstring for Controls"""
    def __init__(self, src_id):
        self.src_id = src_id
        control_dict = extract_controls(self.src_id)
        for c in control_dict:
            self[c] = Control(control_dict[c])

    def update_from_device(self):
        update_from_device(self)

    def load_defaults(self):
        for c in self.itervalues():
            c.set_val(c.default)





class Cam():
    """a simple class that only contains info about a camera"""
    def __init__(self,c):
        self.dict = c
        self.src_id = c['src_id']
        self.name = c['name']
        self.manufacurer = None
        self.serial = c['serial']


class Camera_List(list):
    """docstring for Camera_List"""

    def __init__(self):
        for c in list_devices():
            self.append(Cam(c))


if __name__ == '__main__':
    uvc_cameras = Camera_List()
    for c in uvc_cameras:
        print c.name
        controls = Controls(c.src_id)
        controls.load_defaults()
        controls.update_from_device()
        for c in controls.itervalues():
            if c.flags == "active":
                print c.name, " "*(40-len(c.name)), c.value,c.type, c.min,c.max,c.step
