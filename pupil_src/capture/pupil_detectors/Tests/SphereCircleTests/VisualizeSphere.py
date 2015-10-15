
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



if __name__ == '__main__':
  print "done"


  visualizer = VisualizeSphere()


  visualizer.open_window()
  sphere_radius = 1.0
  sphere = ( (0,0,0),sphere_radius ) # center, radius


  theta_circle_center = pi/2
  phi_circle_center =  pi/4

  points = get_circle_test_points( (theta_circle_center, phi_circle_center), pi/4.0,400, 1.0)
  result = testHaversine((theta_circle_center, phi_circle_center), pi/4.0, 400, 1.0)


  theta = result[0]
  phi =  result[1]
  r_circle = result[2]
  x = sphere_radius * sin(theta) * cos(phi)
  y = sphere_radius * sin(theta) * sin(phi)
  z = sphere_radius * cos(theta)
  while True:
    visualizer.update_window( sphere , points, ( (x,y,z), (x,y,z), r_circle ))
