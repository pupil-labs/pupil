
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
from trackball import Trackball
from math import *
from circleFitUtils import *

class VisualizeSphere(Visualizer):

    def __init__(self):
        super(VisualizeSphere,self).__init__( 800, "Debug Sphere",  True )
        self.running = True

    def update_window(self, eye, points, found_circle ):

        if self._window != None:
            glfwMakeContextCurrent(self._window)

        self.clear_gl_screen()
        self.trackball.push()

        self.draw_coordinate_system(4)

        eye_position = eye[0]
        eye_radius = eye[1]
        # 1. in anthromorphic space, draw pupil sphere and circles on it
       # glLoadMatrixf(self.get_anthropomorphic_matrix())

        self.draw_sphere(eye_position,eye_radius)
        if points:
          draw_points( points, size = 3 , color=RGBA(1.,0,0,1.0), sharpness=1.0 )


        self.draw_circle( found_circle[0], found_circle[2], found_circle[1] )

        self.trackball.pop()

        glfwSwapBuffers(self._window)
        glfwPollEvents()
        return True

    def on_key(self,window, key, scancode, action, mods):
        # self.gui.update_button(button,action,mods)
        super(VisualizeSphere, self).on_key(window, key, scancode, action, mods)
        if key == GLFW_KEY_ESCAPE:
          self.running = False
          self.close_window()



if __name__ == '__main__':
  print "done"


  visualizer = VisualizeSphere()


  visualizer.open_window()
  sphere_radius = 1.0
  sphere = ( (0,0,0),sphere_radius ) # center, radius

  phi_circle_center = pi/2
  theta_circle_center =  pi/2

  right_z = sphere_radius * sin(phi_circle_center) * cos(theta_circle_center)
  right_x = sphere_radius * sin(phi_circle_center) * sin(theta_circle_center)
  right_y = sphere_radius * cos(phi_circle_center)
  print(right_x,right_y,right_z)

  points = get_circle_test_points( (phi_circle_center, theta_circle_center), pi/16,60, 0.4, 0.01)
  #result = testHaversine((phi_circle_center, theta_circle_center), pi/8, 220, 0.3, initial_guess)
  result = testPlanFit((phi_circle_center, theta_circle_center), pi/16, 60, 0.4,  0.01 )

  x = result[0]
  y = result[1]
  z = result[2]

  if( z < 0 ):
    x = -x
    y = -y
    z = -z

  print (x,y,z)
  while visualizer.running:
    visualizer.update_window( sphere , points, ( (x,y,z), (x,y,z), 1 ))
