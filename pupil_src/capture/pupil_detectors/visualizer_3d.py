'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from visualizer import Visualizer
from OpenGL.GL import *
from pyglui.cygl import utils as glutils
from pyglui.cygl.utils import RGBA
from gl_utils.trackball import Trackball
import numpy as np
import math


def get_perpendicular_vector(v):
    """ Finds an arbitrary perpendicular vector to *v*."""
    # http://codereview.stackexchange.com/questions/43928/algorithm-to-get-an-arbitrary-perpendicular-vector
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        logger.error('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array((1, 0, 0))
    if v[1] == 0:
        return np.array((0, 1, 0))
    if v[2] == 0:
        return np.array((0, 0, 1))

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    return np.array([1, 1, -1.0 * (v[0] + v[1]) / v[2]])

class Eye_Visualizer(Visualizer):
    def __init__(self,g_pool , focal_length ):
        super().__init__(g_pool , "Debug Visualizer", False)

        self.focal_length = focal_length
        self.image_width = 640 # right values are assigned in update
        self.image_height = 480

        camera_fov = math.degrees(2.0 * math.atan( self.image_height / (2.0 * self.focal_length)))
        self.trackball = Trackball(camera_fov)

    ############## MATRIX FUNCTIONS ##############################

    def get_anthropomorphic_matrix(self):
        temp =  np.identity(4)
        temp[2,2] *= -1
        return temp


    def get_adjusted_pixel_space_matrix(self,scale):
        # returns a homoegenous matrix
        temp = self.get_anthropomorphic_matrix()
        temp[3,3] *= scale
        return temp

    def get_image_space_matrix(self,scale=1.):
        temp = self.get_adjusted_pixel_space_matrix(scale)
        temp[1,1] *=-1 #image origin is top left
        temp[0,3] = -self.image_width/2.0
        temp[1,3] = self.image_height/2.0
        temp[2,3] = -self.focal_length
        return temp.T

    def get_pupil_transformation_matrix(self,circle_normal,circle_center, circle_scale = 1.0):
        """
            OpenGL matrix convention for typical GL software
            with positive Y=up and positive Z=rearward direction
            RT = right
            UP = up
            BK = back
            POS = position/translation
            US = uniform scale

            float transform[16];

            [0] [4] [8 ] [12]
            [1] [5] [9 ] [13]
            [2] [6] [10] [14]
            [3] [7] [11] [15]

            [RT.x] [UP.x] [BK.x] [POS.x]
            [RT.y] [UP.y] [BK.y] [POS.y]
            [RT.z] [UP.z] [BK.z] [POS.Z]
            [    ] [    ] [    ] [US   ]
        """
        temp = self.get_anthropomorphic_matrix()
        right = temp[:3,0]
        up = temp[:3,1]
        back = temp[:3,2]
        translation = temp[:3,3]
        back[:] = np.array(circle_normal)
        back[2] *=-1 #our z axis is inverted

        if np.linalg.norm(back) != 0:
            back[:] /= np.linalg.norm(back)
            right[:] = get_perpendicular_vector(back)/np.linalg.norm(get_perpendicular_vector(back))
            up[:] = np.cross(right,back)/np.linalg.norm(np.cross(right,back))
            right[:] *= circle_scale
            back[:] *=circle_scale
            up[:] *=circle_scale
            translation[:] = np.array(circle_center)
            translation[2] *= -1
        return   temp.T

    ############## DRAWING FUNCTIONS ##############################

    def draw_debug_info(self, result  ):
        models = result['models']
        eye = models[0]['sphere'];
        direction = result['circle'][1];
        pupil_radius = result['circle'][2];

        status = ' Eyeball center : X: %.2fmm Y: %.2fmm Z: %.2fmm\n Pupil direction:  X: %.2f Y: %.2f Z: %.2f\n Pupil Diameter: %.2fmm\n  ' \
        %(eye[0][0], eye[0][1],eye[0][2],
        direction[0], direction[1],direction[2], pupil_radius*2)

        self.glfont.push_state()
        self.glfont.set_color_float( (0,0,0,1) )

        self.glfont.draw_multi_line_text(5,20,status)


        #draw model info for each model
        delta_y = 20
        for model in models:
            modelStatus =   ('Model: %d \n' %  model['model_id'] ,
                            '    age: %.1fs\n' %(self.g_pool.get_timestamp()-model['birth_timestamp']) ,
                            '    maturity: %.3f\n' % model['maturity'] ,
                            '    solver fit: %.6f\n' % model['solver_fit'] ,
                            '    confidence: %.6f\n' % model['confidence'] ,
                            '    performance: %.6f\n' % model['performance'] ,
                            '    perf.Grad.: %.3e\n' % model['performance_gradient'] ,
                            )
            modeltext = ''.join( modelStatus )
            self.glfont.draw_multi_line_text(self.window_size[0] - 200 ,delta_y, modeltext)

            delta_y += 160

        self.glfont.pop_state()

    def draw_circle(self, circle_center, circle_normal, circle_radius, color=RGBA(1.1,0.2,.8), num_segments = 20):
        vertices = []
        vertices.append( (0,0,0) )  # circle center

        #create circle vertices in the xy plane
        for i in np.linspace(0.0, 2.0*math.pi , num_segments ):
            x = math.sin(i)
            y = math.cos(i)
            z = 0
            vertices.append((x,y,z))

        glPushMatrix()
        glMatrixMode(GL_MODELVIEW )
        glLoadMatrixf(self.get_pupil_transformation_matrix(circle_normal,circle_center, circle_radius))
        glutils.draw_polyline((vertices),color=color, line_type = GL_TRIANGLE_FAN) # circle
        glutils.draw_polyline( [ (0,0,0), (0,0, 4) ] ,color=RGBA(0,0,0), line_type = GL_LINES) #normal
        glPopMatrix()

    def update_window(self, g_pool, result ):

        if not result:
            return

        if not self.window:
            return

        self.begin_update_window()

        self.image_width , self.image_height = g_pool.capture.frame_size

        latest_circle = result['circle']
        predicted_circle = result['predicted_circle']
        edges =  result['edges']
        sphere_models = result['models']

        self.clear_gl_screen()
        self.trackball.push()

        # 2. in pixel space draw video frame
        glLoadMatrixf(self.get_image_space_matrix(15))
        g_pool.image_tex.draw( quad=((0,self.image_height),(self.image_width,self.image_height),(self.image_width,0),(0,0)) ,alpha=0.5)

        glLoadMatrixf(self.get_adjusted_pixel_space_matrix(15))
        self.draw_frustum( self.image_width, self.image_height, self.focal_length )

        glLoadMatrixf(self.get_anthropomorphic_matrix())
        model_count = 0;
        sphere_color = RGBA( 0,147/255.,147/255.,0.2)
        initial_sphere_color = RGBA( 0,147/255.,147/255.,0.2)

        alternative_sphere_color = RGBA( 1,0.5,0.5,0.05)
        alternative_initial_sphere_color = RGBA( 1,0.5,0.5,0.05)

        for model in sphere_models:
            bin_positions = model['bin_positions']
            sphere = model['sphere']
            initial_sphere = model['initial_sphere']

            if model_count == 0:
                # self.draw_sphere(initial_sphere[0],initial_sphere[1], color = sphere_color )
                self.draw_sphere(sphere[0],sphere[1],  color = initial_sphere_color )
                glutils.draw_points(bin_positions, 3 , RGBA(0.6,0.0,0.6,0.5) )

            else:
                #self.draw_sphere(initial_sphere[0],initial_sphere[1], color = alternative_sphere_color )
                self.draw_sphere(sphere[0],sphere[1],  color = alternative_initial_sphere_color )

            model_count += 1


        self.draw_circle( latest_circle[0], latest_circle[1], latest_circle[2], RGBA(0.0,1.0,1.0,0.4))
        # self.draw_circle( predicted_circle[0], predicted_circle[1], predicted_circle[2], RGBA(1.0,0.0,0.0,0.4))

        glutils.draw_points(edges, 2 , RGBA(1.0,0.0,0.6,0.5) )

        glLoadMatrixf(self.get_anthropomorphic_matrix())
        self.draw_coordinate_system(4)

        self.trackball.pop()

        self.draw_debug_info(result)

        self.end_update_window()

        return True


    ############ window callbacks #################
    def on_resize(self,window,w, h):
        Visualizer.on_resize(self,window,w, h)
        self.trackball.set_window_size(w,h)

    def on_char(self,window,char):
        if char == ord('r'):
            self.trackball.distance = [0,0,-0.1]
            self.trackball.pitch = 0
            self.trackball.roll = 0


    def on_scroll(self,window,x,y):
        self.trackball.zoom_to(y)
