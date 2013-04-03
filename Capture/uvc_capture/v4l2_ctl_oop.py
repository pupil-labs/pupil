from v4l2_ctl import *

class Control(dict):
    """docstring for uvcc_Control"""
    def __init__(self,c):
        for key,val in c.items()
            self[key] = val

        self.name = self['name']
        self.current = self['val']
        self.min = self['min']
        self.max = self['max']
        self.step    = self['step']
        self.default = self['def']
        if 'menu' in self:
            self.menu = self[menu]
        else:
            self.menu=None
        self.type  = self['type']
        if 'flags' in self:
            self.flags = self['flags']
        else:
            self.flags = "active"


    def get_val(self):
        return self['val']

    def set_val(self,val):
        set(self['src'],self['name'],val)
        self['val'] = val


class Camera(object):
    """docstring for uvcc_camera"""
    def __init__(self,c):
        self.dict = c
        self.cv_id = c['src_id']
        self.name = c['name']
        self.manufacurer = None
        self.serial = c['serial']

    def init_controls(self):
        control_dict = extract_controls(self.cv_id)

        self.controls = {}
        for c in control_dict:
            self.controls[c] = Control(control_dict[c])

    def load_defaults(self):
        pass

    def update_from_device(self):
        pass

class Camera_List(list):
    """docstring for Camera_List"""

    def __init__(self):
        for c in list_devices():
            self.append(Camera(c))


    def release(self):
        """
        call when done with class instance
        """
        pass



if __name__ == '__main__':
    uvc_cameras = Camera_List()
    for cam in uvc_cameras:
        print cam.name
        cam.init_controls()
        cam.load_defaults()
        for c in cam.controls.itervalues():
            if c.flags == "active":
                print c.name, " "*(40-len(c.name)), c.current,c.type, c.min,c.max,c.step
    uvc_cameras.release()
