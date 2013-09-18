from gl_utils import draw_gl_point_norm
from plugin import Plugin


class Display_Gaze(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool,atb_pos):
        super(Display_Gaze, self).__init__()
        self.g_pool = g_pool
        self.atb_pos = atb_pos
        self.pupil_display_list = []

    def update(self,frame,recent_pupil_positions):
         for pt in recent_pupil_positions:
            if pt['norm_gaze'] is not None:
                self.pupil_display_list.append(pt)
                if len(self.pupil_display_list)>3:
                    self.pupil_display_list.pop(0)

    def gl_display(self):
        for pt in self.pupil_display_list:
            draw_gl_point_norm(pt['norm_gaze'],color=(1.,.2,.2,0.5))

