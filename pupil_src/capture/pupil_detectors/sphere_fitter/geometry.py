import solve
import numpy as np
import logging
logger = logging.getLogger(__name__)


class Circle3D(object):
    def __init__(self,center = np.zeros((3)),normal=np.zeros((3)),radius =0):
        self.center = np.asarray(center).reshape(3)
        self.normal = np.asarray(normal).reshape(3)
        self.radius = radius

    def __str__(self):
        return "Circle(center = %s, normal = %s, radius = %.3f)"%(self.center,self.normal,self.radius)


    def project_to_ellipse(self,intrinsics):
        """
        Construct cone with circle as base and vertex v = (0,0,0).

        For the circle,
            |p - c|^2 = r^2 where (p-c).n = 0 (i.e. on the circle plane)

        A cone is basically concentric circles, with center on the line c->v.
        For any point p, the corresponding circle center c' is the intersection
        of the line c->v and the plane through p normal to n. So,

            d = ((p - v).n)/(c.n)
            c' = d c + v

        The radius of these circles decreases linearly as you approach 0, so

            |p - c'|^2 = (r*|c' - v|/|c - v|)^2

        Since v = (0,0,0), this simplifies to

            |p - (p.n/c.n)c|^2 = (r*|(p.n/c.n)c|/|c|)^2

            |(c.n)p - (p.n)c|^2         / p.n \^2
            ------------------- = r^2 * | --- |
                  (c.n)^2               \ c.n /

            |(c.n)p - (p.n)c|^2 - r^2 * (p.n)^2 = 0

        Expanding out p, c and n gives

            |(c.n)x - (x*n_x + y*n_y + z*n_z)c_x|^2
            |(c.n)y - (x*n_x + y*n_y + z*n_z)c_y|   - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
            |(c.n)z - (x*n_x + y*n_y + z*n_z)c_z|

              ((c.n)x - (x*n_x + y*n_y + z*n_z)c_x)^2
            + ((c.n)y - (x*n_x + y*n_y + z*n_z)c_y)^2
            + ((c.n)z - (x*n_x + y*n_y + z*n_z)c_z)^2
            - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0

              (c.n)^2 x^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*x*c_x + (x*n_x + y*n_y + z*n_z)^2 c_x^2
            + (c.n)^2 y^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*y*c_y + (x*n_x + y*n_y + z*n_z)^2 c_y^2
            + (c.n)^2 z^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*z*c_z + (x*n_x + y*n_y + z*n_z)^2 c_z^2
            - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0

              (c.n)^2 x^2 - 2*(c.n)*c_x*(x*n_x + y*n_y + z*n_z)*x
            + (c.n)^2 y^2 - 2*(c.n)*c_y*(x*n_x + y*n_y + z*n_z)*y
            + (c.n)^2 z^2 - 2*(c.n)*c_z*(x*n_x + y*n_y + z*n_z)*z
            + (x*n_x + y*n_y + z*n_z)^2 * (c_x^2 + c_y^2 + c_z^2 - r^2)

              (c.n)^2 x^2 - 2*(c.n)*c_x*(x*n_x + y*n_y + z*n_z)*x
            + (c.n)^2 y^2 - 2*(c.n)*c_y*(x*n_x + y*n_y + z*n_z)*y
            + (c.n)^2 z^2 - 2*(c.n)*c_z*(x*n_x + y*n_y + z*n_z)*z
            + (|c|^2 - r^2) * (n_x^2*x^2 + n_y^2*y^2 + n_z^2*z^2 + 2*n_x*n_y*x*y + 2*n_x*n_z*x*z + 2*n_y*n_z*y*z)

        Collecting conicoid terms gives

              [xyz]^2 : (c.n)^2 - 2*(c.n)*c_[xyz]*n_[xyz] + (|c|^2 - r^2)*n_[xyz]^2
           [yzx][zxy] : - 2*(c.n)*c_[yzx]*n_[zxy] - 2*(c.n)*c_[zxy]*n_[yzx] + (|c|^2 - r^2)*2*n_[yzx]*n_[zxy]
                      : 2*((|c|^2 - r^2)*n_[yzx]*n_[zxy] - (c,n)*(c_[yzx]*n_[zxy] + c_[zxy]*n_[yzx]))
                [xyz] : 0
                    1 : 0
        """
        c = self.center
        n = self.normal
        r = self.radius
        focal_length = abs(intrinsics[1,1])
        cn = np.dot(n,c)
        c2r2 = np.dot(c,c) - r**2
        ABC = cn**2 - 2.0*cn*c*n + c2r2*n**2
        F = 2*(c2r2*n[1]*n[2] - cn*(n[1]*c[2] + n[2]*c[1]))
        G = 2*(c2r2*n[2]*n[0] - cn*(n[2]*c[0] + n[0]*c[2]))
        H = 2*(c2r2*n[0]*n[1] - cn*(n[0]*c[1] + n[1]*c[0]))

        conic = Conic(ABC[0],H,ABC[1],G*focal_length,F*focal_length,ABC[2]*focal_length**2)
        ellipse = Ellipse.from_conic(conic)
        ellipse.center = np.asarray([ellipse.center[0] + intrinsics[0,2], -ellipse.center[1] + intrinsics[1,2]]) #shift ellipse center and mirror y
        ellipse.angle = -ellipse.angle%np.pi #mirror y
        return ellipse

class Conic(object):
    def __init__(self,a,b,c,d,e,f):
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.E = e
        self.F = f


    @classmethod
    def from_ellipse(cls, e):
        #extracting information from e
        ax = np.cos(e.angle)
        ay = np.sin(e.angle)
        a2 = e.major_radius**2
        b2 = e.minor_radius**2
        A = (ax*ax)/a2 + (ay*ay)/b2
        B = 2*(ax*ay)/a2 - 2*(ax*ay)/b2
        C = (ay*ay)/a2 +(ax*ax)/b2
        D = (-2*ax*ay*e.center[1] - 2*ax*ax*e.center[0])/a2 + (2*ax*ay*e.center[1] - 2*ay*ay*e.center[0])/b2
        E = (-2*ax*ay*e.center[0] - 2*ay*ay*e.center[1])/a2 + (2*ax*ay*e.center[0] - 2*ax*ax*e.center[1])/b2
        F = (2*ax*ay*e.center[0]*e.center[1]+ax*ax*e.center[0]*e.center[0]+ay*ay*e.center[1]*e.center[1])/a2+ (-2*ax*ay*e.center[0]*e.center[1]+ ay*ay*e.center[0]*e.center[0]+ax*ax*e.center[1]*e.center[1])/b2-1
        return cls(A,B,C,D,E,F)

    def __str__(self):
        return "Conic {  %s x^2  + %s xy +  %s y^2 +  %s x +  %s y  + %s  = 0 }"%(self.A,self.B,self.C,self.D,self.E,self.F)


class Conicoid:
    def __init__(self,A=0.0,B=0.0,C=0.0,F=0.0,G=0.0,H=0.0,U=0.0,V=0.0,W=0.0,D=0.0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.F = F
        self.G = G
        self.H = H
        self.U = U
        self.V = V
        self.W = W

    @classmethod
    def from_conic(cls, conic,vertex):
        alpha = vertex[0]
        beta = vertex[1]
        gamma = vertex[2]
        A = gamma**2*conic.A
        B = gamma**2*conic.C
        C = alpha**2*conic.A + alpha*beta*conic.B + beta**2*conic.C + conic.D*alpha + conic.E*beta + conic.F
        F = -gamma * (conic.C * beta + conic.B / 2 * alpha + conic.E / 2)
        G = -gamma * (conic.B / 2 * beta + conic.A * alpha + conic.D / 2)
        H = gamma**2*conic.B/2
        U = gamma**2*conic.D/2
        V = gamma**2*conic.E/2
        W = -gamma * (conic.E / 2 * beta + conic.D / 2 * alpha + conic.F)
        D = gamma**2*conic.F
        return cls(A=A,B=B,C=C,F=F,G=G,H=H,U=U,V=V,W=W,D=D)

    def __str__(self):
        return "Conicoid { %s x^2 + %s y^2 +  %s z^2 + %s yz + 2 %s zx +2 %s xy + %s x + 2 %s y + 2 %s z + %s  = 0 }"%(self.A, self.B, self.C, self.F, self.G, self.H, self.U, self.V, self.W, self.D)


class Ellipse:

    def __init__(self,center=[0,0],major_radius=0.0,minor_radius=0.0,angle=0.0):
        self.center = np.asarray(center).reshape(2)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.angle = angle%np.pi

    @classmethod
    def from_ellipse_dict(cls, e_dict):
        center = np.asarray(e_dict['center'])
        a,b = e_dict['axes']
        if a > b:
            major_radius = a/2
            minor_radius = b/2
            angle = e_dict['angle']*np.pi/180
        else:
            major_radius = b/2
            minor_radius = a/2
            angle = (e_dict['angle']+90)*np.pi/180
        return cls(center,major_radius,minor_radius,angle)

    @classmethod
    def from_conic(cls, conic):
        angle = 0.5*np.arctan2(conic.B,conic.A - conic.C)
        cost = np.cos(angle)
        sint = np.sin(angle)
        cos_squared = cost**2
        sin_squared = sint**2

        Ao = conic.F
        Au = conic.D*cost + conic.E*sint
        Av = -conic.D*sint + conic.E*cost
        Auu= conic.A*cos_squared +conic.C*sin_squared +conic.B*sint*cost
        Avv= conic.A*sin_squared +conic.C*cos_squared -conic.B*sint*cost

        #ROTATED = [Ao Au Av Auu Avv]
        tuCenter = -Au / (2.0*Auu)
        tvCenter = -Av / (2.0*Avv)
        wCenter = Ao - Auu*tuCenter**2 - Avv*tvCenter**2

        center = np.array([tuCenter*cost - tvCenter*sint,tuCenter*sint + tvCenter*cost])
        major_radius = np.sqrt(abs(-wCenter/Auu))
        minor_radius = np.sqrt(abs(-wCenter/Avv))

        if (major_radius < minor_radius):
            major_radius,minor_radius = minor_radius,major_radius
            angle = angle + np.pi/2

        return cls(center,major_radius,minor_radius,angle)

    def __str__(self):
        return "Ellipse(center = %s, major_radius = %.3f, minor_radius = %.3f, angle = %.3f)"%(self.center,self.major_radius,self.minor_radius,self.angle)

    def scale(self,scale):
        self.center *=scale
        self.major_radius *= scale
        self.minor_radius *= scale

    def pointAlongEllipse(self, theta):
        #theta is the angle
        xt = self.center[0] + self.major_radius*np.cos(self.angle)*np.cos(theta) - self.minor_radius*np.sin(self.angle)*np.sin(theta)
        yt = self.center[1] + self.major_radius*np.sin(self.angle)*np.cos(theta) + self.major_radius*np.cos(self.angle)*np.sin(theta)
        return [xt,yt]


    def unproject(self,radius, intrinsics):
        # essentially shift ellipse center by the translation cx and cy from camera matrix
        # and feed this back into unproject()
        focal_length = intrinsics[0,0]
        offset_ellipse = Ellipse((self.center - np.array([intrinsics[0,2],intrinsics[1,2]])) * np.array([1,-1]),
                                                self.major_radius,
                                                self.minor_radius,
                                                np.sign(intrinsics[1,1])*self.angle)
        return unproject(offset_ellipse,radius,focal_length)


class Line:
    def __init__(self, origin, direction):
        self.origin = np.asarray(origin).reshape(-1)
        self.direction = np.asarray(direction).reshape(-1)
        self.direction = self.direction/np.sqrt(sum(self.direction**2))

    def __str__(self):
        return "Line(origin = %s, direction = %s )" %(self.origin,self.direction)


    ## other functions from the eigen class that exist, but may not be used
    #def distance(self,point):
    #    # the distance of a point p to its projection onto the line
    #    pass
    #def intersection_hyperplane(self,hyperplane):
    #    # the parameter value of intersection between this and given hyperplane
    #    pass
    #def intersection_point(self, hyperplane):
    #    # returns parameter value of intersection between this and given hyperplane
    #    pass
    #def projection(self,point):
    #    # returns projection of a point onto the line
    #    pass
    #def pointAt(self,x):
    #    # point at x along this line
    #    pass

class Line2D(Line):
    pass

class Line3D(Line):
    pass


class PupilParams(): #was a structure in C
    def __init__(self, theta = 0, psi = 0, radius = 0):
        self.theta = theta
        self.psi = psi
        self.radius = radius

    def __str__(self):
        return "PupilParams(theta = %s, psi = %s, radius = %s)" %(self.theta, self.psi, self.radius)

class Sphere:
    def __init__(self,center,radius):
        self.center = np.array(center).reshape(3)
        self.radius = radius

    def __str__(self):
        return "Sphere(center = %s, radius = %s)" %(self.center,self.radius)

    def project(self, intrinsics):
        center = project_point(self.center,intrinsics)
        radius = abs(self.radius/self.center[2] * intrinsics[1,1]) #scale based on fx in camera intrinsic matrix
        return Ellipse(center,radius,radius,0)

# extrinsics = np.matrix('1 0 0 0 ; 0 1 0 0 ; 0 0 1 0')

# def project_point(point, intrinsics):
#     point = np.append(np.asarray(point),[1]) #convert point to homogeneous coordinates
#     h_point = point.reshape((4,1))
#     projected_pt = intrinsics * extrinsics * h_point
#     projected_pt = (projected_pt/projected_pt[-1])[:-1] #convert back to cartesian
#     return np.asarray(projected_pt).reshape(2)

def project_point(point, intrinsics):
    x = intrinsics[0,0]*point[0]/point[2] + intrinsics[0,2]
    y = intrinsics[1,1]*point[1]/point[2] + intrinsics[1,2]
    return np.array([x,y])

def unproject_point(point,z,intrinsics):
    return np.array([(point[0]-intrinsics[0,2]) * z / intrinsics[0,0],
                    (point[1]-intrinsics[1,2]) * z / intrinsics[1,1],z])


def unproject(ellipse,circle_radius, focal_length):

    """ TO DO : CASE OF SEEING CIRCLE, DO TRIVIAL CALCULATION (currently wrong result) """
    conic = Conic.from_ellipse(ellipse)
    cam_center_in_ellipse = np.array([[0],[0],[-focal_length]])
    pupil_cone = Conicoid.from_conic(conic = conic, vertex = cam_center_in_ellipse)
    #pupil_cone.initialize_conic(conic,cam_center_in_ellipse) #this step is fine

    a = pupil_cone.A
    b = pupil_cone.B
    c = pupil_cone.C
    f = pupil_cone.F
    g = pupil_cone.G
    h = pupil_cone.H
    u = pupil_cone.U
    v = pupil_cone.V
    w = pupil_cone.W
    d = pupil_cone.D

    """ Get canonical conic form:

        lambda(1) X^2 + lambda(2) Y^2 + lambda(3) Z^2 = mu
        Safaee-Rad 1992 eq (6)
        Done by solving the discriminating cubic (10)
        Lambdas are sorted descending because order of roots doesn't
        matter, and it later eliminates the case of eq (30), where
        lambda(2) > lambda(1)
    """
    lamb = solve.solve_four(1.,
        -(a + b + c),
        (b*c + c*a + a*b - f*f - g*g - h*h),
        -(a*b*c + 2 * f*g*h - a*f*f - b*g*g - c*h*h) )
    if (lamb[0] < lamb[1]):
        logger.error("Lambda 0 > Lambda 1, die")
        return
    if (lamb[1] <= 0):
        logger.error("Lambda 1 > 0, die")
        return
    if (lamb[2] >= 0):
        logger.error("Lambda 2 < 0, die")
        return
    #Calculate l,m,n of plane
    n = np.sqrt((lamb[1] - lamb[2])/(lamb[0]-lamb[2]))
    m = 0.0
    l = np.sqrt((lamb[0] - lamb[1])/(lamb[0]-lamb[2]))

    #Safaee-Rad 1992 Eq 12
    t1 = (b - lamb)*g - f*h
    t2 = (a - lamb)*f - g*h
    t3 = -(a - lamb)*(t1/t2)/g - h/g

    #Safaee-Rad 1992 Eq 8
    mi = 1 / np.sqrt(1 + np.square(t1 / t2) + np.square(t3))
    li = (t1 / t2) * mi
    ni = t3 * mi

    #If li,mi,ni follow the left hand rule, flip their signs
    li = np.reshape(li,(3,))
    mi = np.reshape(mi,(3,))
    ni = np.reshape(ni,(3,))

    if (np.dot(np.cross(li,mi),ni) < 0):
        li = -li
        mi = -mi
        ni = -ni

    T1 = np.zeros((3,3))
    T1[:,0] = li
    T1[:,1] = mi
    T1[:,2] = ni
    T1 = np.asmatrix(T1.T)

    #Calculate t2 a translation transformation from the canonical
    #conic frame to the image space in the canonical conic frame
    #Safaee-Rad 1992 eq (14)
    temp = -(u*li + v*mi + w*ni) / lamb
    T2 = [[temp[0]],[temp[1]],[temp[2]]]
    solutions = [] #two solutions for the circles that we will return

    for i in (1,-1):
        l *= i

        gaze = T1 * np.matrix([[l],[m],[n]])

        #calculate t3, rotation from frame where Z is circle normal

        T3 = np.zeros((3,3))
        if (l == 0):
            if (n == 1):
                logger.error("Warning: l == 0")
                break
            T3 = np.matrix([[0,-1,0],
                [1,0,0],
                [0,0,1]])
        else:
            T3 = np.matrix([[0.,-n*np.sign(l),l],
                [np.sign(l),0.,0.],
                [0.,abs(l),n]]) #changed from round down to abs()

        #calculate circle center
        #Safaee-Rad 1992 eq (38), using T3 as defined in (36)
        lamb =  np.reshape(lamb,(3,))
        T30 = np.array([T3[0,0]**2,T3[1,0]**2,T3[2,0]**2 ])
        T31 = np.array([ [T3[0,0]*T3[0,2]], [T3[1,0]*T3[1,2]] , [T3[2,0]*T3[2,2]] ]) #good
        T32 = np.array([ [T3[0,1]*T3[0,2]], [T3[1,1]*T3[1,2]] , [T3[2,1]*T3[2,2]] ]) #good
        T33 = np.array([T3[0,2]**2 ,T3[1,2]**2 ,T3[2,2]**2 ])

        A = np.dot(lamb,T30)
        B = np.dot(lamb,T31) #good
        C = np.dot(lamb,T32) #good
        D = np.dot(lamb,T33)

        # Safaee-Rad 1992 eq 41
        center_in_Xprime = np.zeros((3,1))
        center_in_Xprime[2] = A*circle_radius/ np.sqrt(B**2 + C**2 - A*D)
        center_in_Xprime[0] = -B / A * center_in_Xprime[2]
        center_in_Xprime[1] = -C / A * center_in_Xprime[2]

        # Safaee-Rad 1992 eq 34
        T0 = [[0],[0],[focal_length]]

        # Safaee-Rad 1992 eq 42 using eq 35
        center = T0+T1*(T2+T3*center_in_Xprime)

        if (center[2] < 0):
            center_in_Xprime = -center_in_Xprime
            center = T0+T1*(T2+T3*center_in_Xprime) #make sure z is positive

        gaze = np.reshape(gaze,(3,))

        if (np.dot(gaze,center) > 0):
            gaze = -gaze
        gaze = gaze/np.linalg.norm(gaze) #normalizing
        gaze = np.array([gaze[0,0],gaze[0,1],gaze[0,2]]) #making it 3 instead of 3x1

        center = np.reshape(center,3)
        center = np.array([center[0,0],center[0,1],center[0,2]]) #making it 3 instead of 3x1

        solutions.append(Circle3D(center,gaze,circle_radius))

    return solutions


if __name__ == '__main__':
    k = np.matrix('100 0 10; 0 -100 10; 0 0 1')

    # print k[0,2]
    p3 = unproject_point((0.0 , 20),100,k)
    p2 = project_point(p3,k)
    print p3,p2
    
    #testing uproject
    ellipse = Ellipse((0.,0.),2.0502,1.0001,2.01)
    circ = ellipse.unproject(1,k)
    print circ[0]
    print circ[1]
    print circ[0].project_to_ellipse(k)
    print circ[1].project_to_ellipse(k)
