'''
Created on Mar 29, 2011

@author: ari
'''
import math

class Vector:
    def __init__(self, coords):
        self.coords = coords

    def __repr__(self):
        return self.coords.__repr__()

    def __add__(self, other):
        return Vector([a+b for a,b in zip(self.coords, other.coords)])

    def __sub__(self, other):
        return Vector([a-b for a,b in zip(self.coords, other.coords)])

    def __mul__(self, scale):
        return Vector([scale*a for a in self.coords])

    def __div__(self, scale):
        return Vector([a/scale for a in self.coords])

    def __rmul__(self, val):
        if val.__class__ == Matrix:
            res = []
            for row in val.entries:
                res.append(self.dot(Vector(row)))
            return Vector(res)
        else:
            return self*val

    def dot(self, other):
        return sum([a*b for a,b in zip(self.coords, other.coords)])

    def __xor__(self, other):
        return Vector([self[1]*other[2]-self[2]*other[1],
                       self[2]*other[0]-self[0]*other[2],
                       self[0]*other[1]-self[1]*other[0]])

    def mag(self):
        return math.sqrt(self.dot(self))

    def __getitem__(self, index):
        return self.coords[index]

    def normalize(self):
        return self/self.mag()

    def cos_angle(self, other):
        res = self.dot(other)/(self.mag()*other.mag())
        return max([-1,min([res, 1])])

    def cot_angle(self, other):
        cos_val = self.cos_angle(other)
        return cos_val/(math.sqrt(1-cos_val**2))

    def angle(self, other):
        return math.acos(self.cos_angle(other))

class Matrix:
    def __init__(self, entries):
        self.entries = entries


