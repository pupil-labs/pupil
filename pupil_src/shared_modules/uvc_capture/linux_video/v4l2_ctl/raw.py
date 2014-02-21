'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
v4l2-ctl is a shell ultity that comes with 4vl2-utils:
it allows getting and setting controls of UVC-capture devices
This is a somewhat hacky interface to v4l2-ctl through subprocess calls

controls are extracted and saved in a dict. Each control is a dict itself.
controls:
    name1:
            min: 0
            max: 255
            ...
            type: "int" or "bool" or "menu"
            menu:
                menu_entry1: 0
                menu_entry2: 1
            src: opencv device id as in /dev/video[src_id]
    name2:

"""
import sys,os
import subprocess as sp
#logging
import logging
logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    # we are running in a |PyInstaller| bundle
    v4l2_ctl = os.path.join(sys._MEIPASS,'v4l2-ctl')
else:
    # we are running in a normal Python environment
    v4l2_ctl = "v4l2-ctl"


# device = "-d"+str(device_number)


def set(device_number,control,value):
    """
    set a value
    open loop
    no error checking
    """
    value = int(value) # turn bools into ints
    device = "-d"+str(device_number)
    sp.Popen([v4l2_ctl,device,"-c"+control+"="+str(value)])

def get(device_number,control):
    """
    get a single control value
    """
    device = "-d"+str(device_number)
    logger.debug("getting control: %s %s %s"%(device_number,control,value))
    ret = sp.check_output([v4l2_ctl,device,"-C"+control])
    return int(ret.split(":")[-1])

def getter_from_device(control):
    """
    get a value
    """
    #print "getter",control
    device = "-d"+ str(control["src"])
    control_name = control["name"]
    ret = sp.check_output([v4l2_ctl,device,"-C"+control_name])
    return int(ret.split(":")[-1])

def getter(control):
    """
    this is a fake getter, it just pulls data from the dict that contains the last changed value on the host
    no control is actually read from the device (because that is to slow)
    use getter_from_device() or get() instead
    """
    return int(control["value"])

def setter(val,control):
    """
    set a value
    open loop
    no error checking
    """
    #print "setter", control,val
    value = int(val) # turn bools into ints
    device = "-d"+str(control["src"])
    control_name = control["name"]
    control["value"]=val
    sp.Popen([v4l2_ctl,device,"-c"+control_name+"="+str(value)])

def update_from_device(controls):
    """
    update all control values
    """
    control_names = [c for c in controls]
    control_string = ",".join(control_names)
    src = controls[control_names[0]]["src"]
    device = "-d"+str(src)

    try:
        ret = sp.check_output([v4l2_ctl,device,"-C"+control_string])
    except:
        #print "could not update uvc controls on device:",device, "please try again"
        return

    for line in ret.split("\n"):
        try:
            name,val = line.split(":")
            controls[name]["value"] = int(val)
        except:
            pass

def extract_controls(device_number):
    """
    using v4l2-ctl -d{device} -l
    we extract all accessible controls
    phrase the response
    example lines:
                  exposure_auto (menu)   : min=0 max=3 default=3 value=3
                                        1: Manual Mode
                                        3: Aperture Priority Mode
              exposure_absolute (int)    : min=3 max=2047 step=1 default=166 value=166 flags=inactive
         exposure_auto_priority (bool)   : default=0 value=1
                   pan_absolute (int)    : min=-36000 max=36000 step=3600 default=0 value=0
                  tilt_absolute (int)    : min=-36000 max=36000 step=3600 default=0 value=0
                 focus_absolute (int)    : min=0 max=255 step=5 default=60 value=60 flags=inactive
                     focus_auto (bool)   : default=1 value=1
                  zoom_absolute (int)    : min=1 max=5 step=1 default=1 value=1

    return a list with a dict for every control
    fields: name, type (menue, int, bool), all other keywords and values are extracted form the response
    """
    device = "-d"+str(device_number)
    try:
        ret = sp.check_output([v4l2_ctl,device,"-L"])
    except:
        logger.warning("v4l2-ctl not found. No uvc control panel will be added")
        return []

    lines = ret.split("\n")
    controls = {}
    control_order = 0
    line = lines.pop(0) #look at first line
    while lines:
        words = line.split(" ")
        words = [word for word in words if len(word)>0] #get rid of remaining spaces
        control = {}
        try:
            control_name = words.pop(0)
        except IndexError:
            ###With some camera this can fail if the line is empty...
            break

        if control_name == "error":
            break

        control["type"] =  words.pop(0)
        colon = words.pop(0) #throw away..
        while words:
            name, value = words.pop(0).split("=")
            try:
                control[name] = int(value)
            except: #sometimes there are some non int flags
                    control[name] = value
        if control["type"] == "(menu)": # in this case the next lines are the menue entries, but we dont know how many...
            control["menu"]  = dict()
            while lines:
                menu_line = lines.pop(0)
                if "(" not in menu_line: #menu line dont contain parents...
                    value,name = menu_line.split(":")
                    control["menu"][name]= int(value)
                else:
                    line = menu_line # end of menue continue with the outer loop
                    break
        else:
            line = lines.pop(0) #take next line  BUG: dont we ignore the last line this way?

        #once done we add the device id and control name and add control to the controls dict
        control["type"] = control["type"][1:-1] #strip of the brackets
        control["name"] = control_name
        control["src"] = device_number
        control["order"] = control_order
        control_order += 1
        controls[control_name] = control
    return controls


def list_devices():
    """
    output:
    HD Webcam C525 (usb-0000:02:03.0-1):
    /dev/video0

    Microsoft LifeCam HD-6000 for (usb-0000:02:03.0-2):
    /dev/video1

    """
    try:
        ret = sp.check_output([v4l2_ctl,"--list-devices"])
    except:
        return []

    paragraphs = ret.split("\n\n")
    devices = []
    for p in paragraphs:
        if p:
            device = {}
            dev_str,loc= p.split(":\n")
            dev,serial = dev_str[:-20], dev_str[-20:]
            serial = serial[1:-1]
            src = int(loc[-1])
            loc = loc.replace("\t","")
            device["name"]=dev
            device["serial"]=serial
            device["location"]=loc
            device["src_id"]=src
            devices.append(device)

    # let make sure we dont have cames with the same name...
    names = [d["name"] for d in devices]
    for idx in range(len(devices))[::-1]:
        dub_count = names[:idx].count(devices[idx]['name'])
        if dub_count:
            devices[idx]['name'] += "(%i)"%dub_count

    return devices

if __name__ == '__main__':
    print list_devices()
    # print extract_controls(0)