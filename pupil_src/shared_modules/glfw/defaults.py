from glfw import *


if __name__ == '__main__':
    import sys


    parameters = {
        GLFW_ACCELERATED : "accelerated" ,
        GLFW_RED_BITS : "red bits" ,
        GLFW_GREEN_BITS : "green bits" ,
        GLFW_BLUE_BITS : "blue bits" ,
        GLFW_ALPHA_BITS : "alpha bits" ,
        GLFW_DEPTH_BITS : "depth bits" ,
        GLFW_STENCIL_BITS : "stencil bits" ,
        GLFW_REFRESH_RATE : "refresh rate" ,
        GLFW_ACCUM_RED_BITS : "accum red bits" ,
        GLFW_ACCUM_GREEN_BITS : "accum green bits" ,
        GLFW_ACCUM_BLUE_BITS : "accum blue bits" ,
        GLFW_ACCUM_ALPHA_BITS : "accum alpha bits" ,
        GLFW_AUX_BUFFERS : "aux buffers" ,
        GLFW_STEREO : "stereo" ,
        GLFW_FSAA_SAMPLES : "FSAA samples" ,
        GLFW_OPENGL_VERSION_MAJOR : "OpenGL major" ,
        GLFW_OPENGL_VERSION_MINOR : "OpenGL minor" ,
        GLFW_OPENGL_FORWARD_COMPAT : "OpenGL forward compatible" ,
        GLFW_OPENGL_DEBUG_CONTEXT : "OpenGL debug context" ,
        GLFW_OPENGL_PROFILE : "OpenGL profile" }

    if not glfwInit():
        sys.exit()
    if not glfwOpenWindow( 0,0,0,0,0,0,0,0, GLFW_WINDOW ):
        glfwTerminate()
        print 'Failed to open GLFW default window'
        sys.exit()
    print "Window size: %ix%i" % glfwGetWindowSize()
    for name,desc in parameters.items():
        print "%s: %i" % (desc, glfwGetWindowParam(name))
    glfwCloseWindow()
    glfwTerminate()

