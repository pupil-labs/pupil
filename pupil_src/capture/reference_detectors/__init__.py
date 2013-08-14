'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

# make shared modules available across pupil_src
if __name__ == '__main__':
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath


# import detector classes from sibling files
from screen_marker_calibration import Screen_Marker_Calibration
from manual_marker_calibration import Manual_Marker_Calibration
from natural_features_calibration import Natural_Features_Calibration
from camera_intrinsics_estimation import Camera_Intrinsics_Estimation


name_by_index = [   'Screen Marker Calibration',
                    'Manual Marker Calibration',
                    'Natural Features Calibration',
                    'Camera Intrinsics Estimation']

detector_by_index = [   Screen_Marker_Calibration,
                        Manual_Marker_Calibration,
                        Natural_Features_Calibration,
                        Camera_Intrinsics_Estimation]

index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
detector_by_name = dict(zip(name_by_index,detector_by_index))



'''
from plugin import Plugin

class Ref_Detector_Template(Plugin):
    """
    template of reference detectors class
    build a detector with this as your template.

    Your derived class needs to have interfaces
    defined by these methods:
    you NEED to do at least what is done in these fn-prototypes

    """
    def __init__(self, global_calibrate, shared_pos, screen_marker_pos, screen_marker_state, atb_pos=(0,0)):
        Plugin.__init__(self)

        self.active = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_pos = shared_pos
        self.shared_screen_marker_pos = screen_marker_pos
        self.shared_screen_marker_state = screen_marker_state
        self.screen_marker_state = -1
        # indicated that no pos has been found
        self.shared_pos = 0,0


        # Creating an ATB Bar required Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "A_Unique_Name", label="",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 150))
        self._bar.add_button("  begin calibrating  ", self.start)

    def start(self):
        self.global_calibrate.value = True
        self.shared_pos[:] = 0,0
        self.active = True

    def stop(self):
        self.global_calibrate.value = False
        self.shared_pos[:] = 0,0
        self.screen_marker_state = -1
        self.active = False


    def update(self,frame):
        if self.active:
            img = frame.img
        else:
            pass

    def __del__(self):
        self.stop()

'''



if __name__ == '__main__':
    pass