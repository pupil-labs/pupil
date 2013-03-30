"""
v4l2-ctl is a shell ultity that comes with 4vl2-utils:
it allows getting and setting controls of UVC-capture devices
This is a somewhat hacky interface to v4l2-ctl through shell calls
"""


import subprocess as sp
v4l2_ctl = "v4l2-ctl"
# device = "-d"+str(device_number)
set_control = "-c"
get_control = "-C"
list_control = "-L"

def set(device_number,control,value):
    """
    set a value
    open loop
    no error checking
    """
    value = int(value) # turn bools into ints
    device = "-d"+str(device_number)
    sp.Popen([v4l2_ctl,device,set_control+control+"="+str(value)])

def get(device_number,control):
    """
    get a value
    """
    device = "-d"+str(device_number)
    print "getting control:", device_number,control,value
    ret = sp.check_output(["v4l2-ctl",device,get_control+control])
    return int(ret.split(":")[-1])

def get_from_device(data):
    """
    get a value
    """
    #print "getter",data
    device = "-d"+data["device"]
    control = data["name"]
    ret = sp.check_output(["v4l2-ctl",device,get_control+control])
    return int(ret.split(":")[-1])

def getter(data):
    """
    this is a fake getter, it just pulls data from the dict that contains the last changed value on the host
    no data is actually read from the device (because that is to slow)
    use get_from_device() instead
    """
    return int(data["value"])

def setter(val,data):
    """
    set a value
    open loop
    no error checking
    """
    #print "setter", data,val
    value = int(val) # turn bools into ints
    device = "-d"+data["device"]
    control=data["name"]
    data["value"]=val
    sp.Popen([v4l2_ctl,device,set_control+control+"="+str(value)])


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
        ret = sp.check_output(["v4l2-ctl",device,list_control])
    except:
        print "v4l2-ctl not found. No uvc control panel will be added"
        return []

    lines = ret.split("\n")
    controls = []
    line = lines.pop(0) #look at first line
    while lines:
        words = line.split(" ")
        words = [word for word in words if len(word)>0] #get rid of remaining spaces
        control = dict()
        control["name"] = words.pop(0)
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
            line = lines.pop(0) #take next line
        controls.append(control)
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
        ret = sp.check_output(["v4l2-ctl","--list-devices"])
    except:
        return []
    print ret
    lines = ret.split("\n")

