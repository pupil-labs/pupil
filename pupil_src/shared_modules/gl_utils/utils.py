'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import OpenGL
from glfw import glfwGetFramebufferSize,glfwGetWindowSize
# OpenGL.FULL_LOGGING = True
OpenGL.ERROR_LOGGING = False
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D
from shader import Shader

import numpy as np



__all__ =  ['make_coord_system_norm_based',
            'make_coord_system_pixel_based',
            'draw_gl_texture',
            'create_named_texture',
            'draw_named_texture',
            'draw_gl_points_norm',
            'draw_gl_point',
            'draw_gl_point_norm',
            'draw_gl_checkerboards',
            'draw_gl_points',
            'draw_gl_polyline',
            'draw_gl_polyline_norm',
            'adjust_gl_view',
            'clear_gl_screen',
            'basic_gl_setup',
            'cvmat_to_glmat']


def cvmat_to_glmat(m):
    mat = np.eye(4,dtype=np.float32)
    mat = mat.flatten()
    # convert to OpenGL matrix
    mat[0]  = m[0,0]
    mat[4]  = m[0,1]
    mat[12] = m[0,2]
    mat[1]  = m[1,0]
    mat[5]  = m[1,1]
    mat[13] = m[1,2]
    mat[3]  = m[2,0]
    mat[7]  = m[2,1]
    mat[15] = m[2,2]
    return mat


def basic_gl_setup():
    glEnable( GL_POINT_SPRITE )
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE) # overwrite pointsize
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glClearColor(1.,1.,1.,0.)

def clear_gl_screen():
    glClear(GL_COLOR_BUFFER_BIT)

def adjust_gl_view(w,h,window):
    """
    adjust view onto our scene.
    """
    h = max(h,1)
    w = max(w,1)

    hdpi_factor = glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0]
    w,h = w*hdpi_factor,h*hdpi_factor
    glViewport(0, 0, w, h)
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # nRange = 1.0
    # if w <= h:
    #     glOrtho(-nRange, nRange, -nRange*h/w, nRange*h/w, -nRange, nRange)
    # else:
    #     glOrtho(-nRange*w/h, nRange*w/h, -nRange, nRange, -nRange, nRange)
    # # switch back to Modelview
    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()

def draw_gl_polyline((positions),color,type='Loop',thickness=1):
    glLineWidth(thickness)
    glColor4f(*color)
    if type=='Loop':
        glBegin(GL_LINE_LOOP)
    elif type=='Strip':
        glBegin(GL_LINE_STRIP)
    elif type=='Polygon':
        glBegin(GL_POLYGON)
    else:
        glBegin(GL_LINES)
    for x,y in positions:
        glVertex3f(x,y,0.0)
    glEnd()

def draw_gl_polyline_norm((positions),color,type='Loop',thickness=1):

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, 1, 0, 1) # gl coord convention
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    draw_gl_polyline(positions,color,type,thickness)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


# def draw_gl_point((x,y),size=20,color=(1.,0.5,0.5,.5)):
#     glColor4f(*color)
#     glPointSize(int(size))
#     glBegin(GL_POINTS)
#     glVertex3f(x,y,0.0)
#     glEnd()


simple_pt_shader = None

def draw_gl_points(points,size=20,color=(1.,0.5,0.5,.5)):
    global simple_pt_shader # we cache the shader because we only create it the first time we call this fn.
    if not simple_pt_shader:

        # we just draw single points, a VBO is much slower than this. But this is a little bit hacked.
        #someday we should replace all legacy fn with vbo's and shaders...
        # shader defines
        VERT_SHADER = """
        #version 120
        varying vec4 f_color;
        void main () {
               gl_Position = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xy,1.,1.);
               gl_PointSize = gl_Vertex.z; //this needs to be used on some hardware we cheat and use the z coord
               f_color = gl_Color;
               }
        """

        FRAG_SHADER = """
        #version 120
        varying vec4 f_color;
        void main()
        {
            float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
            gl_FragColor = mix(f_color, vec4(f_color.rgb,0.0), smoothstep(0.35, 0.5, dist));
        }
        """
        #shader link and compile
        simple_pt_shader = Shader(VERT_SHADER,FRAG_SHADER)

    simple_pt_shader.bind()
    glColor4f(*color)
    glBegin(GL_POINTS)
    for pt in points:
        glVertex3f(pt[0],pt[1],size)
    glEnd()
    simple_pt_shader.unbind()


simple_checkerboard_shader = None

def draw_gl_checkerboards(points,size=60,color=(1.,0.5,0.5,.5), grid=[7.0,7.0]):
    global simple_checkerboard_shader # we cache the shader because we only create it the first time we call this fn.
    if not simple_checkerboard_shader:
        grid = np.array(grid)
        # step = size/grid
        # in this example the step would be 10

        # we just draw single points, a VBO is much slower than this. But this is a little bit hacked.
        #someday we should replace all legacy fn with vbo's and shaders...
        # shader defines
        VERT_SHADER = """
        #version 120
        varying vec4 f_color;
        void main () {
               gl_Position = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xy,1.,1.);
               gl_PointSize = gl_Vertex.z; //this needs to be used on some hardware we cheat and use the z coord
               f_color = gl_Color;
               }
        """

        FRAG_SHADER = """
        #version 120
        varying vec4 f_color;
        uniform vec2 grid;
        void main()
        {
            // get the lowest integer value for the grid
            float total = floor(gl_PointCoord.x*grid.x) + floor(gl_PointCoord.y*grid.y);
            // make the checkerboard by alternating colors
            bool isEven = mod(total,2.0)==0.0;
            vec4 col1 = vec4(0.0,0.0,0.0,1.0);
            vec4 col2 = vec4(1.0,1.0,1.0,1.0);
            gl_FragColor = (isEven)? col1:col2;
        }
        """
        #shader link and compile
        simple_checkerboard_shader = Shader(VERT_SHADER,FRAG_SHADER)

    simple_checkerboard_shader.bind()
    simple_checkerboard_shader.uniformf('grid', *grid)
    glColor4f(*color)
    glBegin(GL_POINTS)
    for pt in points:
        glVertex3f(pt[0],pt[1],size)
    glEnd()
    simple_checkerboard_shader.unbind()



def draw_gl_point_norm(pos,size=20,color=(1.,0.5,0.5,.5)):
    draw_gl_points_norm([pos],size,color)

def draw_gl_point(pos,size=20,color=(1.,0.5,0.5,.5)):
    draw_gl_points([pos],size,color)

def draw_gl_points_norm(pos,size=20,color=(1.,0.5,0.5,.5)):

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, 1, 0, 1) # gl coord convention
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    draw_gl_points(pos,size,color)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def create_named_texture(image):

    if type(image) == tuple:
        height, width, channels = image
        image=None
    else:
        height, width, channels = image.shape

    if  channels == 3:
        gl_blend = GL_BGR
        gl_blend_init = GL_RGB
    else:
        gl_blend = GL_BGRA
        gl_blend_init = GL_RGBA

    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    # Create Texture
    glTexImage2D(GL_TEXTURE_2D,
                        0,
                        gl_blend_init,
                        width,
                        height,
                        0,
                        gl_blend,
                        GL_UNSIGNED_BYTE,
                        image)

    return texture_id

def draw_named_texture(texture_id, image=None, interpolation=True, quad=((0.,0.),(1.,0.),(1.,1.),(0.,1.))):
    """
    We draw the image as a texture on a quad from 0,0 to img.width,img.height.
    We set the coord system to pixel dimensions.
    to save cpu power, update can be false and we will reuse the old img instead of uploading the new.
    """

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glEnable(GL_TEXTURE_2D)

    # update texture
    if image is not None:
        height, width, channels = image.shape
        if  channels == 3:
            gl_blend = GL_BGR
        else:
            gl_blend = GL_BGRA

        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexSubImage2D(GL_TEXTURE_2D,
                            0,
                            0,
                            0,
                            width,
                            height,
                            gl_blend,
                            GL_UNSIGNED_BYTE,
                            image)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) # interpolation here
    if not interpolation:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    # someday replace with this:
    # glEnableClientState(GL_VERTEX_ARRAY)
    # glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    # Varray = numpy.array([[0,0],[0,1],[1,1],[1,0]],numpy.float)
    # glVertexPointer(2,GL_FLOAT,0,Varray)
    # glTexCoordPointer(2,GL_FLOAT,0,Varray)
    # indices = [0,1,2,3]
    # glDrawElements(GL_QUADS,1,GL_UNSIGNED_SHORT,indices)
    glColor4f(1.0,1.0,1.0,1.0)
    # Draw textured Quad.
    glBegin(GL_QUADS)
    # glTexCoord2f(0.0, 0.0)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(*quad[0])
    glTexCoord2f(1.0, 1.0)
    glVertex2f(*quad[1])
    glTexCoord2f(1.0, 0.0)
    glVertex2f(*quad[2])
    glTexCoord2f(0.0, 0.0)
    glVertex2f(*quad[3])
    glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)



def draw_gl_texture(image,interpolation=True):
    """
    We draw the image as a texture on a quad from 0,0 to img.width,img.height.
    Simple anaymos texture one time use. Look at named texture fn's for better perfomance
    """


    height, width, channels = image.shape
    if  channels == 3:
        gl_blend = GL_BGR
        gl_blend_init = GL_RGB
    else:
        gl_blend = GL_BGRA
        gl_blend_init = GL_RGBA

    glPixelStorei(GL_UNPACK_ALIGNMENT,1)

    glEnable(GL_TEXTURE_2D)

    # Create Texture and upload data
    glTexImage2D(GL_TEXTURE_2D,
                        0,
                        gl_blend_init,
                        width,
                        height,
                        0,
                        gl_blend,
                        GL_UNSIGNED_BYTE,
                        image)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) # interpolation here
    if not interpolation:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


    # someday replace with this:
    # glEnableClientState(GL_VERTEX_ARRAY)
    # glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    # Varray = numpy.array([[0,0],[0,1],[1,1],[1,0]],numpy.float)
    # glVertexPointer(2,GL_FLOAT,0,Varray)
    # glTexCoordPointer(2,GL_FLOAT,0,Varray)
    # indices = [0,1,2,3]
    # glDrawElements(GL_QUADS,1,GL_UNSIGNED_SHORT,indices)
    glColor4f(1.0,1.0,1.0,1.0)
    # Draw textured Quad.
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(0,0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(1,0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(1,1)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0,1)
    glEnd()

    glDisable(GL_TEXTURE_2D)


def make_coord_system_pixel_based(img_shape):
    height,width,channels = img_shape
    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, height, 0) # origin in the top left corner just like the img np-array
    # Switch back to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def make_coord_system_norm_based():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 1, 0, 1) # gl coord convention
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


