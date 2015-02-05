'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Gaze_Mapping_Plugin
import dill


class Dummy_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Dummy_Gaze_Mapper"""
    def __init__(self, g_pool):
        super(Dummy_Gaze_Mapper, self).__init__(g_pool)

    def update(self,frame,events):
        gaze_pts = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_pts.append({'norm_pos':p['norm_pos'][:],'confidence':p['confidence'],'timestamp':p['timestamp']})

        events['gaze'] = gaze_pts

    def get_init_dict(self):
        return {}


class Simple_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Simple_Gaze_Mapper"""
    def __init__(self, g_pool,map_fn):
        super(Simple_Gaze_Mapper, self).__init__(g_pool)
        # check if we are loading a pickled function or creating for the first time.
        self.map_fn = map_fn
        if isinstance(self.map_fn,str):
            self.map_fn = dill.loads(self.map_fn)
        else:
            # otherwise we know we are init the first time
            pass


    def update(self,frame,events):
        gaze_pts = []

        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_point = self.map_fn(p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp']})

        events['gaze'] = gaze_pts

    def get_init_dict(self):
        # serialize the function
        map_fn_str = dill.dumps(self.map_fn)
        return {'map_fn':map_fn_str}


class Volumetric_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self,g_pool,params):
        super(Volumetric_Gaze_Mapper, self).__init__(g_pool)
        self.params = params

    def update(self,frame,events):
        gaze_pts = []
        raise NotImplementedError
        events['gaze'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}

class Bilateral_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params):
        super(Volumetric_Gaze_Mapper, self).__init__(g_pool)
        self.params = params

    def update(self,frame,events):
        gaze_pts = []
        raise NotImplementedError
        events['gaze'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}