"""
This is a somewhat hacky interface to v4l2-ctl through shell calls
"""


import subprocess as sp
v4l2_ctl = "v4l2-ctl"
# device = "-d"+str(device_number)
set_control = "-c"
get_control = "-C"
list_control = "-l"

def set(device_number,control,value):
    """
    set a value
    open loop
    no error checking
    """
    device = "-d"+str(device_number)
    sp.Popen([v4l2_ctl,device,set_control+control+"="+str(value)])

def get(device_number,control):
    """
    under construction: no errorchecking or conversion
    """
    device = "-d"+str(device_number)
    stdout = sp.check_output([v4l2_ctl,device,get_control+control])
    return stdout

def extract_controls(device_number):
    """
    using v4l2-ctl -d{device} -l
    we extract all accessible controls
    phrase the response
    example lines:
    # exposure_auto (menu)   : min=0 max=3 default=3 value=1
    # exposure_absolute (int)    : min=3 max=2047 step=1 default=166 value=166
    # exposure_auto_priority (bool)   : default=0 value=1

    return a list with a dict for every control
    fields: name, type (menue, int, bool), all other keywords and values are extracted form the response
    """

    try:
        ret = sp.check_output(["v4l2-ctl","-d0","-l"])
    except:
        print "v4l2-ctl not found. No uvc control panel will be added"
        return []

    lines = ret.split("\n")
    controls = []
    for line in lines:
        words = line.split(" ")
        words = [word for word in words if len(word)>0] #get rid of remaining spaces
        if words:
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
            controls.append(control)
    return controls