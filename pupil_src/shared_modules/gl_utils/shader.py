#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
#
# Copyright Tristam Macdonald 2008.
#
# Distributed under the Boost Software License, Version 1.0
# (see http://www.boost.org/LICENSE_1_0.txt)
#
''' Base shader class.

    Example:
    --------
    shader = Shader()

    shader.bind()
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(lut.target, lut.id)
    shader.uniformi('lut', 1)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(texture.target, texture.id)
    shader.uniformi('texture', 0)
    shader.uniformf('pixel', 1.0/texture.width, 1.0/texture.height)

    texture.blit(x,y,w,h)
    shader.unbind()
'''
import os
import OpenGL.GL as gl
import ctypes



class ShaderException(Exception):
    '''The root exception for all shader related errors.'''
    pass

class ShaderVertexException(ShaderException):
    '''Exception for vertex compilation errors'''
    pass

class ShaderFragmentException(ShaderException):
    '''Exception for fragment compilation errors'''
    pass

class ShaderLinkException(ShaderException):
    '''Exception for linking errors'''
    pass



class Shader:
    ''' Base shader class. '''

    def __init__(self, vertex_code = None, fragment_code = None):
        '''
        Compile and link vertex code and fragment code into a shader.

        :Parameters:
            ``vertex_code``: string
                Vertex code

            ``fragment_code``: string
                Fragment code
        '''

        self.uniforms = {}
        self._vertex_code = vertex_code
        self._fragment_code = fragment_code

        # create the program handle
        self.handle = gl.glCreateProgram()

        # we are not linked yet
        self.linked = False

        # create the vertex shader
        self._build_shader(vertex_code, gl.GL_VERTEX_SHADER)

        # create the fragment shader
        self._build_shader(fragment_code, gl.GL_FRAGMENT_SHADER)

        # the geometry shader will be the same, once pyglet supports the
        # extension self.createShader(frag, GL_GEOMETRY_SHADER_EXT) attempt to
        # link the program
        self._link()



    def _build_shader(self, strings, shader_type):
        ''' Actual building of the shader '''

        count = len(strings)
        # if we have no source code, ignore this shader
        if count < 1:
            return

        # create the shader handle
        shader = gl.glCreateShader(shader_type)

        # Upload shader code
        gl.glShaderSource(shader, strings)

        # compile the shader
        gl.glCompileShader(shader)

        # retrieve the compile status
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)

        # if compilation failed, raise exception and print the log
        if not status:
            if shader_type == gl.GL_VERTEX_SHADER:
                raise ShaderException, \
                    'Vertex compilation: ' + gl.glGetShaderInfoLog(shader)
            elif shader_type == gl.GL_FRAGMENT_SHADER:
                raise ShaderException, \
                    'Fragment compilation:' + gl.glGetShaderInfoLog(shader)
            else:
                raise ShaderException, \
                    gl.glGetShaderInfoLog(shader)
        else:
            # all is well, so attach the shader to the program
            gl.glAttachShader(self.handle, shader)

    def _link(self):
        ''' Link the program '''

        gl.glLinkProgram(self.handle)
        # retrieve the link status
        temp = ctypes.c_int(0)
        gl.glGetProgramiv(self.handle, gl.GL_LINK_STATUS, ctypes.byref(temp))

        # if linking failed, print the log
        if not temp:
            # retrieve the log length
            gl.glGetProgramiv(self.handle,
                              gl.GL_INFO_LOG_LENGTH, ctypes.byref(temp))

            # create a buffer for the log
            log = gl.glGetProgramInfoLog(self.handle) #, temp, None, buffer)

            raise ShaderException, 'Linking: '+ log
        else:
            # all is well, so we are linked
            self.linked = True

    def bind(self):
        ''' Bind the program, i.e. use it. '''
        gl.glUseProgram(self.handle)

    def unbind(self):
        ''' Unbind whatever program is currently bound - not necessarily this
            program, so this should probably be a class method instead. '''
        gl.glUseProgram(0)

    def uniformf(self, name, *vals):
        ''' Uploads float uniform(s), program must be currently bound. '''

        loc = self.uniforms.get(name,
                                gl.glGetUniformLocation(self.handle,name))
        #if loc < 0:
        #    raise ShaderException, \
        #        '''Unknow uniform location '%s' ''' % name
        self.uniforms[name] = loc

        # Check there are 1-4 values
        if len(vals) in range(1, 5):
            # Select the correct function
            { 1 : gl.glUniform1f,
              2 : gl.glUniform2f,
              3 : gl.glUniform3f,
              4 : gl.glUniform4f
              # Retrieve uniform location, and set it
            }[len(vals)](loc, *vals)

    def uniformi(self, name, *vals):
        ''' Upload integer uniform(s), program must be currently bound. '''

        loc = self.uniforms.get(name,
                                gl.glGetUniformLocation(self.handle,name))
        #if loc < 0:
        #    raise ShaderException, \
        #        '''Unknow uniform location '%s' ''' % name
        self.uniforms[name] = loc

        # Checks there are 1-4 values
        if len(vals) in range(1, 5):
            # Selects the correct function
            { 1 : gl.glUniform1i,
              2 : gl.glUniform2i,
              3 : gl.glUniform3i,
              4 : gl.glUniform4i
              # Retrieves uniform location, and set it
            }[len(vals)](loc, *vals)


    def uniform_matrixf(self, name, mat):
        ''' Upload uniform matrix, program must be currently bound. '''

        loc = self.uniforms.get(name,
                                gl.glGetUniformLocation(self.handle,name))
        #if loc < 0:
        #    raise ShaderException, \
        #        '''Unknow uniform location '%s' ''' % name
        self.uniforms[name] = loc

        # Upload the 4x4 floating point matrix
        gl.glUniformMatrix4fv(loc, 1, False, (ctypes.c_float * 16)(*mat))


    def get_vertex_code(self, lineno=True):
        code = ''
        for lineno,line in enumerate(self._vertex_code.split('\n')):
            code += '%3d: ' % (lineno+1) + line + '\n'
        return code

    def get_fragment_code(self,lineno=True):
        code = ''
        for lineno,line in enumerate(self._fragment_code.split('\n')):
            code += '%3d: ' % (lineno+1) + line + '\n'
        return code

