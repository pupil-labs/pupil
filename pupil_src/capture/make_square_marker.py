'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cv2
import numpy as np

def encode_marker(mId):
    marker_id_str = "%02d"%mId
    # marker is based on grid of black (0) /white (1) pixels
    #  b|b|b|b|b
    #  b|o|m|o|b    b = black border feature
    #  b|m|m|m|b    o = orientation feature
    #  b|o|m|o|b    m = message feature
    #  b|b|b|b|b
    grid = 5
    m_with_b = np.zeros((grid,grid),dtype=np.uint8)
    m = m_with_b[1:-1,1:-1]

    #bitdepth = grid-border squared - 3 for orientation (orientation still yields one bit)
    bitdepth = ((5-2)**2)-3

    if mId>=(2**bitdepth):
        raise Exception("ERROR: ID overflow, this marker can only hold %i bits of information" %bitdepth)

    msg = [0]*bitdepth
    for i in range(len(msg))[::-1]:
        msg[i] = mId%2
        mId = mId >>1

    # out first bit is encoded in the orientation corners of the marker:
    #               MSB = 0                   MSB = 1
    #               W|*|*|W   ^               B|*|*|B   ^
    #               *|*|*|*  / \              *|*|*|*  / \
    #               *|*|*|*   |  UP           *|*|*|*   |  UP
    #               B|*|*|W   |               W|*|*|B   |

    msb = msg.pop(0)
    if msb:
        orientation  = 0,1,0,0
    else:
        orientation = 1,0,1,1

    m[0,0], m[-1,0], m[-1,-1], m[0,-1] = orientation

    msg_mask = np.ones(m.shape,dtype=np.bool)
    msg_mask[0,0], msg_mask[-1,0], msg_mask[-1,-1], msg_mask[0,-1] = 0,0,0,0
    m[msg_mask] = msg[::-1]

    # print "Marker: \n", m_with_b
    return m_with_b

def write_marker_png(mId,size):
    marker_id_str = "%02d"%mId
    m = encode_marker(mId)*255
    m = cv2.resize(m,(size,size),interpolation=cv2.INTER_NEAREST)
    m = cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('marker '+marker_id_str+'.png',m)


def write_all_markers_png(size):
    rows,cols,m_size = 8,8,7
    canvas = np.ones((m_size*rows,m_size*cols),dtype=np.uint8)
    for mid in range(64):
        marker_with_padding = np.ones((7,7))
        marker_with_padding[1:-1,1:-1] = encode_marker(mid)
        m_size = marker_with_padding.shape[0]
        r = (mid%rows) * m_size
        c = (mid/cols) * m_size
        canvas[r:r+m_size,c:c+m_size] = marker_with_padding*255
    canvas = cv2.resize(canvas,(size,size),interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('all_markers.png',canvas)


if __name__ == '__main__':
    write_all_markers_png(2000)
    write_marker_png(8,800)
