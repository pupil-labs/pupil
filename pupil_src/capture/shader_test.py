if __name__ == '__main__':
    # make shared modules available across pupil_src
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath


import numpy as np
import OpenGL.GL as gl
from gl_utils import VertexBuffer, Shader,adjust_gl_view,clear_gl_screen,basic_gl_setup,draw_gl_point_norm,draw_gl_point
from glfw import *
from OpenGL.GLU import gluOrtho2D



if __name__ == '__main__':

    width = 512
    height = 512
    pos_x = width/2.0
    pos_y = height/2.0



    # Callback functions
    def on_resize(window,w, h):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        glfwMakeContextCurrent(active_window)

    # Initialize glfw
    glfwInit()
    world_window = glfwCreateWindow(width, height, "World", None, None)
    glfwMakeContextCurrent(world_window)

    # Register callbacks world_window
    glfwSetWindowSizeCallback(world_window,on_resize)

    basic_gl_setup()


    # vertex buffer setup
    V = np.array( [ ((-1, -1, 0), (0,0)),
                    ((1, -1, 0), (1,0)),
                    ((1, 1, 0), (1,1)),
                    ((-1, 1, 0), (0,1)) ],
                  dtype = [('position','f4',3),
                           ('tex_coord','f4',2)] )
    I = np.array( [0,1,2, 0,2,3 ], dtype=np.uint32 )
    rect = VertexBuffer(V,I)


    # shader defines
    vertex = """
        void main()
        {
            gl_FrontColor = gl_Color;
            gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;
            gl_TexCoord[0] = gl_MultiTexCoord0;
        }"""

    fragment = """
#ifdef GL_ES
precision mediump float;
#endif

uniform float time;
uniform vec2 mouse;
uniform vec2 resolution;

float pi = atan(1.)*4.;
float m = 1.; // Change the speed easily with this multiplier!

mat2 rotate(float a)
{
    a = mod(a,2.*pi);
    return mat2(cos(a),-sin(a),
            sin(a), cos(a));
}

float clength(vec2 v)
{
    return max(abs(v.x),abs(v.y));
}

void main( void ) {

    vec2 res = vec2(resolution.x/resolution.y,1.0);
    vec2 p = ( gl_FragCoord.xy / resolution.y );
    p -= res/2.;
    p = gl_TexCoord[0].xy - vec2(.5,.5);

    vec2 rp;
    vec3 c;
    float t = 2.2;
    rp = p * rotate(t * m);
    c.r = smoothstep(0.35,0.345,clength(rp))-smoothstep(0.21,0.205,clength(rp));

    rp = p * rotate(t*.75 * m);
    c.g = smoothstep(0.35,0.345,clength(rp))-smoothstep(0.21,0.205,clength(rp));

    rp = p * rotate(t*.5 * m);
    c.b = smoothstep(0.35,0.345,clength(rp))-smoothstep(0.21,0.205,clength(rp));

    gl_FragColor = vec4( c , 0.5 );

    }
        """
    #shader link and compile
    shader = Shader(vertex, fragment)

    while not glfwWindowShouldClose(world_window):
        clear_gl_screen()

        # get normalize curser coords
        # win_size = np.array(glfwGetWindowSize(world_window))
        # win_size *= .5,-.5
        # pos = np.array(glfwGetCursorPos(world_window))/win_size + np.array((-1,1))
        pos =glfwGetCursorPos(world_window)
        # a legacy point
        draw_gl_point((0,0))

        # # Set Projection Matrix
        # gl.glMatrixMode(gl.GL_PROJECTION)
        # gl.glLoadIdentity()
        # gluOrtho2D(0, width, height, 0) # origin in the top left corner just like the img np-array
        # # Switch back to Model View Matrix
        # gl.glMatrixMode(gl.GL_MODELVIEW)
        # gl.glLoadIdentity()

        shader.bind()
        pos = 0,0

        for x in range(300):
            pass
            # call to change shader may be done here:
            # uniformf('name', new_value)
            # # update the vertex we use our shader on
            size = 1.0
            rect.vertices['position'][0] = pos[0]-size, pos[1]-size, 0
            rect.vertices['position'][1] = pos[0]+size, pos[1]-size, 0
            rect.vertices['position'][2] = pos[0]+size, pos[1]+size, 0
            rect.vertices['position'][3] = pos[0]-size, pos[1]+size, 0
            rect.upload()
            rect.draw( gl.GL_TRIANGLES )
        shader.unbind()


        glfwSwapBuffers(world_window)
        glfwPollEvents()


