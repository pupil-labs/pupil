from OpenGL.GL import *
from OpenGL.GLU import *
from numpy.linalg import inv
from math import sqrt
from gl_utils import clear_gl_screen
import cv2
import numpy as np
 
def MTL(filename):
    """
    Load an obj. Only read map_Kd for now
    """
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError, "mtl file doesn't start with newmtl stmt"
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            image = cv2.imread("../ressources/"+mtl['map_Kd'], 1)
            iy, ix, _ = image.shape
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, ix, iy, 0, GL_BGR, GL_UNSIGNED_BYTE, image)
        else:
            #no other effect are read
            pass
    return contents
 
class OBJ:
    def __init__(self, filename, mult, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.mtl = None
        material = None

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = map(float, values[1:4])
                v = [a * mult for a in v]  #mutliplication factor
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL("../ressources/"+values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
 
    def draw(self):
        """
        Will draw the OBJ. It use GL_DEPTH_TEST and Texture
        """
        #glPushMatrix()

        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)
        if self.faces[1][2] == [0, 0, 0]:
            glDisable(GL_TEXTURE_2D)
            
        
        glFrontFace(GL_CCW)
        #don't draw anything that you can't see
        glEnable(GL_DEPTH_TEST)
        """
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE)
        # now glColor* changes diffuse reflection   
        glColor3f(0.2, 0.5, 0.8)
        # draw some objects here  
        glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR)
        # glColor* no longer changes diffuse reflection   
        # now glColor* changes specular reflection   
        glColor3f(0.9, 0.0, 0.2)
        glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR) 
        glColor3f(100., 0., 0.)

        #glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (0., 0., 0., 1.))
        #glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (1., 1., 1., 1.))
        #glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,(0.7, 0.7, 0.7, 1.))

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE) 
        #default
        """

        for face in self.faces:
            vertices, normals, texture_coords, material = face

 
            if self.mtl != None:
                mtl = self.mtl[material]
                if 'texture_Kd' in mtl:
                    # use diffuse texmap
                    glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])

                elif 'kd' in mtl:
                    # just use diffuse colour
                    glColor(mtl['Kd'])
            else:
                glColor3f(1., 1., 1.)
 
            glBegin(GL_TRIANGLES)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()

        #glFlush()    # Render now

        glDisable(GL_TEXTURE_2D)
        #glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_DEPTH_TEST)
        #glPopMatrix()
  
def drawIn2D( A, K, marker ):
    """
    This function allow to draw an obj in 2D. It take 3D point of the obj and project them on the image of the scene.
    The scnene don't need to be previously undistored. But no texture or light effect can be add.
    """

    #get 2D points of each vertices
    imgpts, _ = cv2.projectPoints(np.array(marker['obj'].vertices), marker['rot'], marker['trans'], A, None)
    imgpts = np.int32(imgpts).reshape(-1,2)

    glBegin(GL_TRIANGLES)
    for face in marker['obj'].faces:
        vertices, _, _, _ = face
        for i in range(len(vertices)):
            glVertex2fv(imgpts[vertices[i] - 1])

    glEnd()


def unproject( A, img_point ):
    #doesn't work properly, better use gluUnproject or find an other way to do it
    matPI = np.matrix(img_point)
    matA = np.matrix(A)
    ainv = inv(matA)
    matPC = ainv*matPI.T

    return matPC.T

def calculFrustum( A, K, w, h):
    """
    Calcul the frustum for the camera perspective and the normalized coordinates.
    This function use unproject to find the exact borders.
    """

    img_pts = np.zeros(3)
    frustum = np.zeros(6)

    img_pts[0] = w/2
    img_pts[1] = h/2
    img_pts[2] = 0.0
    real_pts1 = unproject( A, img_pts )
    img_pts[0] = 0.0
    img_pts[1] = h/2
    img_pts[2] = 0.0
    real_pts2 = unproject( A, img_pts )
    frustum[0] = -1.0 * distance(real_pts1.getA(), real_pts2.getA()) # left

    img_pts[0] = w
    img_pts[1] = h/2
    img_pts[2] = 0.0
    real_pts2 = unproject( A, img_pts )
    frustum[1] =  distance(real_pts1.getA(), real_pts2.getA()) # right

    img_pts[0] = w/2
    img_pts[1] = h
    img_pts[2] = 0.0
    real_pts2 = unproject( A, img_pts )
    frustum[2] = -1.0 * distance(real_pts1.getA(), real_pts2.getA()) # bottom

    img_pts[0] = w/2
    img_pts[1] = 0.0
    img_pts[2] = 0.0
    real_pts2 = unproject( A, img_pts )
    frustum[3] =  distance(real_pts1.getA(), real_pts2.getA()) # top

    frustum[4] =  0.1  # Near clipping distance
    frustum[5] = 1000  # Far clipping distance
    
    return frustum

def calculPerpective(A, near, far):
    """
    Calcul the perspective matrix needed in GL_PROJECTION to place the camera correctly 
    using the intrinsic parameter
    """

    persp = np.zeros((4,4))

    persp[0][0] = A[0][0]
    persp[1][1] = A[1][1]
    persp[0][1] = A[0][1]
    persp[0][2] = -A[0][2]
    persp[1][2] = -A[1][2]
    persp[3][2] = -1

    persp[2][2] = near + far
    persp[2][3] = near * far

    return persp


def drawAxis(taille=10):
    """
    Simply draw X, Y and Z axis.
    """
    glBegin(GL_LINES)

    # draw X axis in red
    glColor3f( 1.0, 0.0, 0.0)
    glVertex3f( 0.0, 0.0, 0.0)
    glVertex3f( taille, 0.0, 0.0)

    # draw Y axis in green
    glColor3f( 0.0, 1.0, 0.0)
    glVertex3f( 0.0, 0.0, 0.0)
    glVertex3f( 0.0, taille, 0.0)

    # draw Z axis in blue
    glColor3f( 0.0, 0.0, 1.0)
    glVertex3f( 0.0, 0.0, 0.0)
    glVertex3f( 0.0, 0.0, taille)

    glEnd()


# GtoC = Global to Camera = Mire to Camera
def calculTransformation( R, T):
    """
    calcul the transformation matrix you need to use in ModelView to transform all
    your coordinate from global world to camera world (camera coordinate will be 0, 0, 0)
    """

    #from a 3x1 rotation vector to a 3x3 rotation matrix
    R = cv2.Rodrigues(R)[0]
    
    GtoC = np.zeros([4,4], dtype=np.float32)
    GtoC[0][0] = R[0][0] # colonne 1
    GtoC[1][0] = R[1][0]
    GtoC[2][0] = R[2][0]
    GtoC[3][0] = 0.0

    GtoC[0][1] = R[0][1] # colonne 2
    GtoC[1][1] = R[1][1]
    GtoC[2][1] = R[2][1]
    GtoC[3][1] = 0.0

    GtoC[0][2] = R[0][2] # colonne 3
    GtoC[1][2] = R[1][2]
    GtoC[2][2] = R[2][2]
    GtoC[3][2] = 0.0

    GtoC[0][3] = T[0] # colonne 4
    GtoC[1][3] = T[1]
    GtoC[2][3] = T[2]
    GtoC[3][3] = 1.0

    GtoC = cv2.transpose(GtoC)  #OpenGL is collumn-major, so the matrix need to be inverted

    return GtoC

def invertAxis():
    """
    Most of the time, Y and Z axis are inverted between OpenGL and OpenCV
    This function invert the Y and Z axis for a matrix mode
    """
    cvToGl = np.zeros([4,4], dtype=np.float32)
    cvToGl[0][0] = 1.0
    cvToGl[1][1] = -1.0 # Invert the y axis 
    cvToGl[2][2] = -1.0 # invert the z axis 
    cvToGl[3][3] = 1.0
    glMultMatrixf(cvToGl)


def applyFrustum( A, K, frame_size, roi, near, far ):
    """
    Apply a new glFrustum calculated with the intrinsic parameters.
    The first method to implement unproject to work.
    The second method works well but is less precise than calculate perspective
    """

    #need unproject to work
    #frustum = calculFrustum( A, K, frame_size[1], frame_size[0] )
    #glFrustum(frustum[0], frustum[1], frustum[2], frustum[3], frustum[4], frustum[5])

    # Camera parameters
    fx = A[0][0] # Focal length in x axis
    fy = A[1][1] # Focal length in y axis (usually the same?)
    cx = A[0][2] # Camera primary point x
    cy = A[1][2] # Camera primary point y

    if roi != None:
        screen_width = roi[2] # In pixels
        screen_height = roi[3] # In pixels
    else :
        screen_width = frame_size[1] # In pixels
        screen_height = frame_size[0] # In pixels

    fovY = 1/(fy/screen_height * 2)
    fovX = 1/(fx/screen_width * 2)
    aspectRatio = (screen_width/screen_height) * (fy/fx)

    #frustum_height = near * fovY
    #frustum_width = frustum_height * aspectRatio
    frustum_width = near * fovX * aspectRatio
    frustum_height = near * fovY * aspectRatio

    offset_x = (screen_width/2 - cx)/screen_width * frustum_width * 2
    offset_y = (screen_height/2 - cy)/screen_height * frustum_height * 2

    # Build and apply the projection matrix
    glFrustum(-frustum_width - offset_x, frustum_width - offset_x, -frustum_height - offset_y, frustum_height - offset_y, near, far)



def glDrawFromCamera( A, K, R, T, frame_size, roi= None, obj=None ) :
    """
    Make all changement to draw correctly in the camera world
    """

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()

    near = 0.1  # Near clipping distance
    far = 10000  # Far clipping distance

    #we prefere use glOrtho to convert to normalized device coordinates (NDC) and calcul the perspective transformation 
    #than to use glFrustum because there is no implemented functions good enough to calculate glFrustum
    #applyFrustum( A, K, frame_size, roi, near, far )
    
    #coord need to be rectify only if there is an important gap beetween the image shape and the shape given by camera_matrix
    if (A[0][2] * 2 - frame_size[1]) > 5:
        rectify_x = (A[0][2] * 2) - frame_size[1]
    else:
        rectify_x = 0
    if (A[1][2] * 2 - frame_size[0]) > 5:
        rectify_y = (A[1][2] * 2) - frame_size[0]
    else:
        rectify_y = 0

    #origin is at bottom left
    if roi == None:
        glOrtho(0, frame_size[1], rectify_y, frame_size[0] + rectify_y, near, far)
    else:
        #not the same proportion of croped area between height and width, offset rectify this plus inverse the Y axis between openGL and openCV
        glOrtho(0, roi[2], frame_size[0] - roi[3] + rectify_y,  frame_size[0] + rectify_y, near, far)

    #calcul perspectiv using intrinsic parameter
    #see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/  for more info
    persp = calculPerpective(A, near, far)
    persp = cv2.transpose(persp)  #OpenGL is collumn-major, so the matrix need to be inverted
    glMultMatrixf(persp)          

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    #GLcam looking at +z axis
    gluLookAt( 0., 0., 0.,      # Eye-position
               0., 0., T[2],    # View-point
               0., -1., 0. )    # Up-vector
    #gluLookAt( 0., 0., 0., T[0], T[1], T[2], 0., -1., 0.) #GLcam looking at the marker

    # change to camera world
    GtoC = calculTransformation( R, T )
    glMultMatrixf(GtoC)   

    # draw in world coordinate
    #obj = None
    if obj == None:
        drawAxis(76)
    else:
        obj.draw()

    #return to previous settings
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()


      

def glInit():  #for now, allows only one light, but can be easily modified
    """
    Don't do anything for now, but can initiate light
    """
    #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    # Enables Smooth Color Shading try GL_FLAT for (lack of) fun.
    #glShadeModel(GL_SMOOTH)

    #glEnable(GL_LIGHTING)

    #glLightfv(GL_LIGHT0, GL_POSITION,  (10., 10., 10., 0.))
    #glLightfv(GL_LIGHT0, GL_AMBIENT, (0., 0., 0., 1.))
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, (1., 1., 1., 1.))

    #glEnable(GL_LIGHT0)

    #glFrontFace(GL_CCW)


def setLightPos(pos):
    glLightfv(GL_LIGHT0, GL_POSITION,  pos)


def apply_gl_texture( image ):
    """
    Very close to draw_gl_texture but clean the screen before and don't draw the texture in Z=0
    (needed for GL_DEPTH_TEST)
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

    height, width, channels = image.shape
    data_3 = image

    gl_blend = (None,GL_LUMINANCE,None,GL_BGR,GL_BGRA)[channels]
    gl_blend_init = (None,GL_LUMINANCE,None,GL_RGB,GL_RGBA)[channels]

    #texname = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    glEnable(GL_TEXTURE_2D)
    #glBindTexture(GL_TEXTURE_2D, texname)

    # Create Texture and upload data
    glTexImage2D(GL_TEXTURE_2D, 0, gl_blend_init, width, height, 0, gl_blend, GL_UNSIGNED_BYTE, data_3)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    
    # Draw textured Quad.
    glBegin(GL_QUADS)
    glTexCoord3f(0.0, 1.0, 100.)
    glVertex3f(0.,0., 0.)
    glTexCoord3f(1.0, 1.0, 100.)
    glVertex3f(1.,0., 0.)
    glTexCoord3f(1.0, 0.0, 100.)
    glVertex3f(1.,1., 0.)
    glTexCoord3f(0.0, 0.0, 100.)
    glVertex3f(0.,1., 0.)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
