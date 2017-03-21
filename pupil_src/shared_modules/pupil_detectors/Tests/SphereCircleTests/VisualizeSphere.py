'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import sys , os
 # Make all pupil shared_modules available to this Python session.
pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_detectors', 1)[0] + 'pupil_detectors'
sys.path.append(pupil_base_dir)

# Make all pupil shared_modules available to this Python session.
pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))

from visualizer_3d import Visualizer
from glfw import *
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D

from pyglui.cygl.utils import init
from pyglui.cygl.utils import RGBA
from pyglui.cygl.utils import *
from pyglui.cygl import utils as glutils
from gl_utils import Trackball
from math import *
import random
import time
from SphereCircleTest import *
class VisualizeSphere(Visualizer):

    def __init__(self):
        super().__init__( 800, "Debug Sphere",  True )
        self.running = True

    def update_window(self, eye, points, found_circle ):

        if self._window != None:
            glfwMakeContextCurrent(self._window)

        self.clear_gl_screen()
        self.trackball.push()

        glLoadMatrixf(self.get_anthropomorphic_matrix())
        self.draw_coordinate_system(4)

        eye_position = eye[0]
        eye_radius = eye[1]

        self.draw_sphere(eye_position,eye_radius)
        if points:
          draw_points( points, size = 3 , color=RGBA(1.,0,0,1.0), sharpness=1.0 )


        self.draw_circle( found_circle[0], found_circle[1], found_circle[2], color=RGBA(1.,1,0,0.5) )

        self.trackball.pop()

        glfwSwapBuffers(self._window)
        glfwPollEvents()
        return True

    def on_key(self,window, key, scancode, action, mods):
        # self.gui.update_button(button,action,mods)
        super().on_key(window, key, scancode, action, mods)
        if key == GLFW_KEY_ESCAPE:
          self.running = False
          self.close_window()



if __name__ == '__main__':
  print "done"


  visualizer = VisualizeSphere()


  visualizer.open_window()
  sphere_radius = 1.0
  sphere = ( (0,0,0),sphere_radius ) # center, radius
  points = ()
  circle = ( (0,0,0), (0,0,0), 1 )
  lastTime = time.clock()
  while visualizer.running:

    if time.clock() - lastTime  >= 1.0:

      phi_circle_center = random.uniform(0, pi/2.0)
      theta_circle_center =   random.uniform(-pi/2.0, pi/2.0)

      circle_distortion =  random.uniform(0, 0.2)
      circle_segment_amount = random.uniform(0.2, 1.0)
      circle_point_amount =  random.randint(5, 100)
      circle_opening = random.uniform(pi/32, pi/1.5 )

      right_z = sphere_radius * sin(phi_circle_center) * cos(theta_circle_center)*cos(circle_opening)
      right_x = sphere_radius * sin(phi_circle_center) * sin(theta_circle_center)*cos(circle_opening)
      right_y = sphere_radius * cos(phi_circle_center)*cos(circle_opening)

      print"Set Position: {} {} {}".format( right_x,right_y,right_z)
      print"Set Circle Radius: {}".format( sin(circle_opening) )
      points = get_circle_test_points( (phi_circle_center, theta_circle_center), circle_opening,circle_point_amount, circle_segment_amount, circle_distortion)
      result = testPlanFit( sphere,  (phi_circle_center, theta_circle_center), circle_opening, circle_point_amount, circle_segment_amount,  circle_distortion )

      x = result[0]
      y = result[1]
      z = result[2]
      nx = result[3]
      ny = result[4]
      nz = result[5]
      radius = result[6]
      residual = result[7]

      circle =  ( (x,y,z), (nx,ny,nz), radius )

      print "Fit result Position: {} {} {}".format(x,y,z)
      print "Fit result Radius: {}".format(radius)
      print "Fit result Residual {}".format(residual)
      lastTime = time.clock()

    visualizer.update_window( sphere , points, circle )
