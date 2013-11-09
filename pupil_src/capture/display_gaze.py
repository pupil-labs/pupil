'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from gl_utils import draw_gl_point_norm,Shader,VertexBuffer
from plugin import Plugin
import numpy as np
import OpenGL.GL as gl

from methods import denormalize

class Display_Gaze(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool,atb_pos):
        super(Display_Gaze, self).__init__()
        self.g_pool = g_pool
        self.atb_pos = atb_pos
        self.pupil_display_list = []
        self.vertex_and_shader_setup()

    def update(self,frame,recent_pupil_positions):
        for pt in recent_pupil_positions:
            if pt['norm_gaze'] is not None:
                self.pupil_display_list.append(denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y=True))
        self.pupil_display_list[:-3] = []

    def gl_display(self):
        self.shader.bind()
        for pos in self.pupil_display_list:
            size = 20
            self.vb.vertices['position'] = [pos[0]-size, pos[1]-size, 0],[pos[0]+size, pos[1]-size, 0], [pos[0]+size, pos[1]+size, 0],[pos[0]-size, pos[1]+size, 0]
            # rect.vertices['position'][1] = pos[0]+size, pos[1]-size, 0
            # rect.vertices['position'][2] = pos[0]+size, pos[1]+size, 0
            # rect.vertices['position'][3] = pos[0]-size, pos[1]+size, 0
            self.vb.upload()
            self.vb.draw( gl.GL_TRIANGLES )
        self.shader.unbind()



    def vertex_and_shader_setup(self):
        # vertex buffer setup
        V = np.array( [ ((-1, -1, 0), (0,0)),
                    ((1, -1, 0), (1,0)),
                    ((1, 1, 0), (1,1)),
                    ((-1, 1, 0), (0,1)) ],
                  dtype = [('position','f4',3),
                           ('tex_coord','f4',2)] )
        I = np.array( [0,1,2, 0,2,3 ], dtype=np.uint32 )
        self.vb = VertexBuffer(V,I)


        # shader defines
        vertex = """
            void main()
            {
                gl_FrontColor = gl_Color;
                gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;
                gl_TexCoord[0] = gl_MultiTexCoord0;
            }"""

        fragment = """
            vec4 color = vec4(0.99,0.1,0.4,0.5); //translucent red
            vec4 faded_color = color;
            void main(void)
            {
                faded_color.w = float(0.0);
                
                //gl_TexCoord[0].xy is the interpolated texture coords (relative to the underlying vertex), we often call them u and v
                //gl_FragCoord.xy is absolute window coords in pixelspace
                
                float dist = distance(gl_TexCoord[0].xy, vec2(0.5, 0.5)); // .5 is center
                gl_FragColor = mix(color, faded_color, smoothstep(0.4, 0.49, dist));

               //if (dist < .4) {
               //   gl_FragColor = mix(vec4(0.90, 0.50, 0.40, 1.0), vec4(0.90, 0.50, 0.40, 0.8), smoothstep(0.05, 0.4, dist));
               //}
               //else {
               //  gl_FragColor = mix(vec4(0.90, 0.50, 0.40, 0.8), vec4(0.90, 0.50, 0.40, 0.0), smoothstep(0.4, 0.49, dist));
               //
               //}
            }
            """
        #shader link and compile
        self.shader = Shader(vertex, fragment)