#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import sys
import ctypes
import numpy as np
import OpenGL
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut


class VertexAttribute(object):
    def __init__(self, count, gltype, stride, offset):
        self.count  = count
        self.gltype = gltype
        self.stride = stride
        self.offset = ctypes.c_void_p(offset)



class VertexAttribute_color(VertexAttribute):
    def __init__(self, count, gltype, stride, offset):
        assert count in (3, 4), \
            'Color attributes must have count of 3 or 4'
        VertexAttribute.__init__(self, count, gltype, stride, offset)
    def enable(self):
        gl.glColorPointer(self.count, self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)



class VertexAttribute_edge_flag(VertexAttribute):
    def __init__(self, count, gltype, stride, offset):
        assert count == 1, \
            'Edge flag attribute must have a size of 1'
        assert gltype in (gl.GL_BYTE, gl.GL_UNSIGNED_BYTE, gl.GL_BOOL), \
            'Edge flag attribute must have boolean type'
        VertexAttribute.__init__(self, 1, gltype, stride, offset)
    def enable(self):
        gl.glEdgeFlagPointer(self.stride, self.offset)
        gl.glEnableClientState(gl.GL_EDGE_FLAG_ARRAY)



class VertexAttribute_fog_coord(VertexAttribute):
    def __init__(self, count, gltype, stride, offset):
        VertexAttribute.__init__(self, count, gltype, stride, offset)
    def enable(self):
        gl.glFogCoordPointer(self.count, self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_FOG_COORD_ARRAY)



class VertexAttribute_normal(VertexAttribute):
    def __init__(self, count, gltype, stride, offset):
        assert count == 3, \
            'Normal attribute must have a size of 3'
        assert gltype in (gl.GL_BYTE, gl.GL_SHORT,
                          gl.GL_INT, gl.GL_FLOAT, gl.GL_DOUBLE), \
                                'Normal attribute must have signed type'
        VertexAttribute.__init__(self, 3, gltype, stride, offset)
    def enable(self):
        gl.glNormalPointer(self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)



class VertexAttribute_secondary_color(VertexAttribute):
    def __init__(self, count, gltype, strude, offset):
        assert count == 3, \
            'Secondary color attribute must have a size of 3'
        VertexAttribute.__init__(self, 3, gltype, stride, offset)
    def enable(self):
        gl.glSecondaryColorPointer(3, self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_SECONDARY_COLOR_ARRAY)



class VertexAttribute_tex_coord(VertexAttribute):
    def __init__(self, count, gltype, stride, offset):
        assert gltype in (gl.GL_SHORT, gl.GL_INT, gl.GL_FLOAT, gl.GL_DOUBLE), \
            'Texture coord attribute must have non-byte signed type'
        VertexAttribute.__init__(self, count, gltype, stride, offset)
    def enable(self):
        gl.glTexCoordPointer(self.count, self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)



class VertexAttribute_position(VertexAttribute):
    def __init__(self, count, gltype, stride, offset):
        assert count > 1, \
            'Vertex attribute must have count of 2, 3 or 4'
        assert gltype in (gl.GL_SHORT, gl.GL_INT, gl.GL_FLOAT, gl.GL_DOUBLE), \
            'Vertex attribute must have signed type larger than byte'
        VertexAttribute.__init__(self, count, gltype, stride, offset)
    def enable(self):
        gl.glVertexPointer(self.count, self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)



class VertexAttribute_generic(VertexAttribute):
    def __init__(self, count, gltype, stride, offset, index, normalized=False ):
        assert count in (1, 2, 3, 4), \
            'Generic attributes must have count of 1, 2, 3 or 4'
        VertexAttribute.__init__(self, count, gltype, stride, offset)
        self.index = index
        self.normalized = normalized
    def enable(self):
        gl.glVertexAttribPointer( self.index, self.count, self.gltype,
                                  self.normalized, self.stride, self.offset );
        gl.glEnableVertexAttribArray( self.index )



class VertexBufferException(Exception):
    pass



class VertexBuffer(object):
    ''' '''
    def __init__(self, vertices, indices=None):
        gltypes = { 'float32': gl.GL_FLOAT,
                    'float'  : gl.GL_DOUBLE, 'float64': gl.GL_DOUBLE,
                    'int8'   : gl.GL_BYTE,   'uint8'  : gl.GL_UNSIGNED_BYTE,
                    'int16'  : gl.GL_SHORT,  'uint16' : gl.GL_UNSIGNED_SHORT,
                    'int32'  : gl.GL_INT,    'uint32' : gl.GL_UNSIGNED_INT }
        dtype = vertices.dtype
        names = dtype.names or []
        stride = vertices.itemsize
        offset = 0
        index = 1 # Generic attribute indices starts at 1
        self.attributes = {}
        self.generic_attributes = []
        if indices is None:
            indices = np.arange(vertices.size,dtype=np.uint32)
        for name in names:
            if dtype[name].subdtype is not None:
                gtype = str(dtype[name].subdtype[0])
                count = reduce(lambda x,y:x*y, dtype[name].shape)
            else:
                gtype = str(dtype[name])
                count = 1
            if gtype not in gltypes.keys():
                raise VertexBufferException('Data type not understood')
            gltype = gltypes[gtype]
            if name in['position', 'color', 'normal', 'tex_coord',
                       'fog_coord', 'secondary_color', 'edge_flag']:
                vclass = 'VertexAttribute_%s' % name
                attribute = eval(vclass)(count,gltype,stride,offset)
                self.attributes[name[0]] = attribute
            else:
                attribute = VertexAttribute_generic(count,gltype,stride,offset,index)
                self.generic_attributes.append(attribute)
                index += 1
            offset += dtype[name].itemsize
        self.vertices = vertices
        self.indices = indices
        self.vertices_id = gl.glGenBuffers(1)

        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBufferData( gl.GL_ARRAY_BUFFER, self.vertices, gl.GL_STATIC_DRAW )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )

        self.indices = indices
        self.indices_id = gl.glGenBuffers(1)
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )
        gl.glBufferData( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices, gl.GL_STATIC_DRAW )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 )

    def upload(self):
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBufferData( gl.GL_ARRAY_BUFFER, self.vertices, gl.GL_STATIC_DRAW )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )
        


    def draw( self, mode=gl.GL_QUADS, what='pnctesf' ):
        gl.glPushClientAttrib( gl.GL_CLIENT_VERTEX_ARRAY_BIT )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )
        for attribute in self.generic_attributes:
            attribute.enable()
        for c in self.attributes.keys():
            if c in what:
                self.attributes[c].enable()
        gl.glDrawElements( mode, self.indices.size, gl.GL_UNSIGNED_INT, None)
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )
        gl.glPopClientAttrib( )




# -----------------------------------------------------------------------------
def on_display():
    global cube, theta, phi, frame, time, timebase

    frame += 1
    time = glut.glutGet( glut.GLUT_ELAPSED_TIME )
    if (time - timebase > 1000):
        print frame*1000.0/(time-timebase)
        timebase = time;		
        frame = 0;

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glPushMatrix()
    gl.glRotatef(theta, 0,0,1)
    gl.glRotatef(phi, 0,1,0)
    gl.glDisable( gl.GL_BLEND )
    gl.glEnable( gl.GL_LIGHTING )
    gl.glEnable( gl.GL_DEPTH_TEST )
    gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )
    gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
    cube.draw( gl.GL_QUADS, 'pnc' )
    gl.glDisable( gl.GL_POLYGON_OFFSET_FILL )
    gl.glEnable( gl.GL_BLEND )
    gl.glDisable( gl.GL_LIGHTING )
    gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
    gl.glDepthMask( gl.GL_FALSE )
    gl.glColor( 0.0, 0.0, 0.0, 0.5 )
    cube.draw( gl.GL_QUADS, 'p' )
    gl.glDepthMask( gl.GL_TRUE )
    gl.glPopMatrix()

    glut.glutSwapBuffers()
    
def on_reshape(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode( gl.GL_PROJECTION )
    gl.glLoadIdentity( )
    glu.gluPerspective( 45.0, float(width)/float(height), 2.0, 10.0 )
    gl.glMatrixMode( gl.GL_MODELVIEW )
    gl.glLoadIdentity( )
    gl.glTranslatef( 0.0, 0.0, -5.0 )

def on_keyboard(key, x, y):
    if key == '\033':
        sys.exit()

def on_timer(value):
    global theta, phi
    theta += 0.25
    phi += 0.25
    glut.glutPostRedisplay()
    glut.glutTimerFunc(10, on_timer, 0)

def on_idle():
    global theta, phi
    theta += 0.25
    phi += 0.25
    glut.glutPostRedisplay()


if __name__ == '__main__':
    p = ( ( 1, 1, 1), (-1, 1, 1), (-1,-1, 1), ( 1,-1, 1),
          ( 1,-1,-1), ( 1, 1,-1), (-1, 1,-1), (-1,-1,-1) )
    n = ( ( 0, 0, 1), (1, 0, 0), ( 0, 1, 0),
          (-1, 0, 1), (0,-1, 0), ( 0, 0,-1) );
    c = ( ( 1, 1, 1), ( 1, 1, 0), ( 1, 0, 1), ( 0, 1, 1),
          ( 1, 0, 0), ( 0, 0, 1), ( 0, 1, 0), ( 0, 0, 0) );

    vertices = np.array(
        [ (p[0],n[0],c[0]), (p[1],n[0],c[1]), (p[2],n[0],c[2]), (p[3],n[0],c[3]),
          (p[0],n[1],c[0]), (p[3],n[1],c[3]), (p[4],n[1],c[4]), (p[5],n[1],c[5]),
          (p[0],n[2],c[0]), (p[5],n[2],c[5]), (p[6],n[2],c[6]), (p[1],n[2],c[1]),
          (p[1],n[3],c[1]), (p[6],n[3],c[6]), (p[7],n[3],c[7]), (p[2],n[3],c[2]),
          (p[7],n[4],c[7]), (p[4],n[4],c[4]), (p[3],n[4],c[3]), (p[2],n[4],c[2]),
          (p[4],n[5],c[4]), (p[7],n[5],c[7]), (p[6],n[5],c[6]), (p[5],n[5],c[5]) ], 
        dtype = [('position','f4',3), ('normal','f4',3), ('color','f4',3)] )


    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
    glut.glutCreateWindow("Python VBO")
    glut.glutReshapeWindow(400, 400)
    glut.glutDisplayFunc(on_display)
    glut.glutReshapeFunc(on_reshape)
    glut.glutKeyboardFunc(on_keyboard)
    glut.glutTimerFunc(10, on_timer, 0)
    #glut.glutIdleFunc(on_idle)

    gl.glPolygonOffset( 1, 1 )
    gl.glClearColor(1,1,1,1);
    gl.glEnable( gl.GL_DEPTH_TEST )
    gl.glEnable( gl.GL_COLOR_MATERIAL )
    gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
    gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
    gl.glEnable( gl.GL_LIGHT0 )
    gl.glLight( gl.GL_LIGHT0, gl.GL_DIFFUSE,  (1.0,1.0,1.0,1.0) )
    gl.glLight( gl.GL_LIGHT0, gl.GL_AMBIENT,  (0.1,0.1,0.1,1.0) )
    gl.glLight( gl.GL_LIGHT0, gl.GL_SPECULAR, (0.0,0.0,0.0,1.0) )
    gl.glLight( gl.GL_LIGHT0, gl.GL_POSITION, (0.0,1.0,2.0,1.0) )
    gl.glEnable( gl.GL_LINE_SMOOTH )

    theta, phi = 0, 0
    frame, time, timebase = 0, 0, 0
    cube = VertexBuffer(vertices)

    glut.glutMainLoop()

