'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D
from shader import Shader

def basic_gl_setup():

    glEnable( GL_POINT_SPRITE )
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE) # overwrite pointsize
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glClearColor(1.,1.,1.,0.)

def clear_gl_screen():
    glClear(GL_COLOR_BUFFER_BIT)

def adjust_gl_view(w,h):
    """
    adjust view onto our scene so
    that a quad from 0,0 to w,h fits into it perfecly
    """
    if h == 0:
        h = 1

    if w == 0:
        w = 1

    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    nRange = 1.0
    if w <= h:
        glOrtho(-nRange, nRange, -nRange*h/w, nRange*h/w, -nRange, nRange)
    else:
        glOrtho(-nRange*w/h, nRange*w/h, -nRange, nRange, -nRange, nRange)
    # switch back to Modelview
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def draw_gl_polyline((positions),(r,g,b,a),type='Loop'):
    glColor4f(r,g,b,a)
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

def draw_gl_polyline_norm((positions),(r,g,b,a),type='Loop'):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(-1, 1, -1, 1) # origin at the center positive up, positve right
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    draw_gl_polyline(positions,(r,g,b,a),type)
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
               gl_PointSize = gl_Vertex.z; //this needs to be used on some hardware we cheat ande use the z coord
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


def draw_gl_point_norm(pos,size=20,color=(1.,0.5,0.5,.5)):
    draw_gl_points_norm([pos],size,color)

def draw_gl_point(pos,size=20,color=(1.,0.5,0.5,.5)):
    draw_gl_points([pos],size,color)

def draw_gl_points_norm(pos,size=20,color=(1.,0.5,0.5,.5)):

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(-1, 1, -1, 1) # origin at the center positive up, positve right
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    draw_gl_points(pos,size,color)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

def draw_gl_texture(image,interpolation=True):
    """
    We draw the image as a texture on a quad from 0,0 to img.width,img.height.
    """

    height, width, channels = image.shape
    if  channels == 3:
        gl_blend = GL_BGR
        gl_blend_init = GL_RGB
    else:
        gl_blend = GL_BGRA
        gl_blend_init = GL_RGBA
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
    glEnable(GL_TEXTURE_2D)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) # interpolation here
    if not interpolation:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, height, 0) # origin in the top left corner just like the img np-array
    # Switch back to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

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
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(width, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(width, height)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(0.0, height)
    glEnd()
    glDisable(GL_TEXTURE_2D)

    # # Set Projection Matrix
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # gluOrtho2D(0, width, height, 0) # origin in the top left corner just like the img np-array
    # # Switch back to Model View Matrix
    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()
