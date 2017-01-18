'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from methods import Roi
from pyglui.cygl.utils import draw_points as cygl_draw_points
from pyglui.cygl.utils import RGBA as cygl_rgba
from pyglui.cygl.utils import draw_polyline as cygl_draw_polyline
from OpenGL.GL import GL_LINE_LOOP

class UIRoi(Roi):
    """
    this object inherits from ROI and adds some UI helper functions
    """
    def __init__(self,array_shape):
        super().__init__(array_shape)
        self.max_x = array_shape[1]-1
        self.min_x = 1
        self.max_y = array_shape[0]-1
        self.min_y = 1

        #enforce contraints
        self.lX = max(self.min_x,self.lX)
        self.uX = min(self.max_x,self.uX)
        self.lY = max(self.min_y,self.lY)
        self.uY = min(self.max_y,self.uY)


        self.handle_size = 45
        self.active_edit_pt = False
        self.active_pt_idx = None
        self.handle_color = cygl_rgba(.5,.5,.9,.9)
        self.handle_color_selected = cygl_rgba(.5,.9,.9,.9)
        self.handle_color_shadow = cygl_rgba(.0,.0,.0,.5)

    @property
    def rect(self):
        return [[self.lX,self.lY],
                [self.uX,self.lY],
                [self.uX,self.uY],
                [self.lX,self.uY]]

    def move_vertex(self,vert_idx,pt):
        x,y = pt
        x,y = int(x),int(y)
        x,y = min(self.max_x,x),min(self.max_y,y)
        x,y = max(self.min_x,x),max(self.min_y,y)
        thresh = 45
        if vert_idx == 0:
            x = min(x,self.uX-thresh)
            y = min(y,self.uY-thresh)
            self.lX,self.lY = x,y
        if vert_idx == 1:
            x = max(x,self.lX+thresh)
            y = min(y,self.uY-thresh)
            self.uX,self.lY = x,y
        if vert_idx == 2:
            x = max(x,self.lX+thresh)
            y = max(y,self.lY+thresh)
            self.uX,self.uY = x,y
        if vert_idx == 3:
            x = min(x,self.uX-thresh)
            y = max(y,self.lY+thresh)
            self.lX,self.uY = x,y

    def mouse_over_center(self,edit_pt,mouse_pos,w,h):
        return edit_pt[0]-w/2 <= mouse_pos[0] <=edit_pt[0]+w/2 and edit_pt[1]-h/2 <= mouse_pos[1] <=edit_pt[1]+h/2

    def mouse_over_edit_pt(self,mouse_pos,w,h):
        for p,i in zip(self.rect,range(4)):
            if self.mouse_over_center(p,mouse_pos,w,h):
                self.active_pt_idx = i
                self.active_edit_pt = True
                return True

    def draw(self,ui_scale=1):
        cygl_draw_polyline(self.rect,color=cygl_rgba(.8,.8,.8,0.9),thickness=1,line_type=GL_LINE_LOOP)

    def draw_points(self,ui_scale=1):
        if self.active_edit_pt:
            inactive_pts = self.rect[:self.active_pt_idx]+self.rect[self.active_pt_idx+1:]
            active_pt = [self.rect[self.active_pt_idx]]
            cygl_draw_points(inactive_pts,size=(self.handle_size+10)*ui_scale,color=self.handle_color_shadow,sharpness=0.3)
            cygl_draw_points(inactive_pts,size=self.handle_size*ui_scale,color=self.handle_color,sharpness=0.9)
            cygl_draw_points(active_pt,size=(self.handle_size+30)*ui_scale,color=self.handle_color_shadow,sharpness=0.3)
            cygl_draw_points(active_pt,size=(self.handle_size+10)*ui_scale,color=self.handle_color_selected,sharpness=0.9)
        else:
            cygl_draw_points(self.rect,size=(self.handle_size+10)*ui_scale,color=self.handle_color_shadow,sharpness=0.3)
            cygl_draw_points(self.rect,size=self.handle_size*ui_scale,color=self.handle_color,sharpness=0.9)

