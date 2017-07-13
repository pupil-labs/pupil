'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import csv
from itertools import chain
import logging
from plugin import Analysis_Plugin_Base
from pyglui import ui
# logging
logger = logging.getLogger(__name__)


class Raw_Data_Exporter(Analysis_Plugin_Base):
    '''
    pupil_positions.csv
    keys:
        timestamp - timestamp of the source image frame
        index - associated_frame: closest world video frame
        id - 0 or 1 for left/right eye
        confidence - is an assessment by the pupil detector on how sure we can be on this measurement. A value of `0` indicates no confidence. `1` indicates perfect confidence. In our experience usefull data carries a confidence value greater than ~0.6. A `confidence` of exactly `0` means that we don't know anything. So you should ignore the position data.        norm_pos_x - x position in the eye image frame in normalized coordinates
        norm_pos_x - x position in the eye image frame in normalized coordinates
        norm_pos_y - y position in the eye image frame in normalized coordinates
        diameter - diameter of the pupil in image pixels as observed in the eye image frame (is not corrected for perspective)

        method - string that indicates what detector was used to detect the pupil

        --- optional fields depending on detector

        #in 2d the pupil appears as an ellipse available in `3d c++` and `2D c++` detector
        2d_ellipse_center_x - x center of the pupil in image pixels
        2d_ellipse_center_y - y center of the pupil in image pixels
        2d_ellipse_axis_a - first axis of the pupil ellipse in pixels
        2d_ellipse_axis_b - second axis of the pupil ellipse in pixels
        2d_ellipse_angle - angle of the ellipse in degrees


        #data made available by the `3d c++` detector

        diameter_3d - diameter of the pupil scaled to mm based on anthropomorphic avg eye ball diameter and corrected for perspective.
        model_confidence - confidence of the current eye model (0-1)
        model_id - id of the current eye model. When a slippage is detected the model is replaced and the id changes.

        sphere_center_x - x pos of the eyeball sphere is eye pinhole camera 3d space units are scaled to mm.
        sphere_center_y - y pos of the eye ball sphere
        sphere_center_z - z pos of the eye ball sphere
        sphere_radius - radius of the eyeball. This is always 12mm (the anthropomorphic avg.) We need to make this assumption because of the `single camera scale ambiguity`.

        circle_3d_center_x - x center of the pupil as 3d circle in eye pinhole camera 3d space units are mm.
        circle_3d_center_y - y center of the pupil as 3d circle
        circle_3d_center_z - z center of the pupil as 3d circle
        circle_3d_normal_x - x normal of the pupil as 3d circle. Indicates the direction that the pupil points at in 3d space.
        circle_3d_normal_y - y normal of the pupil as 3d circle
        circle_3d_normal_z - z normal of the pupil as 3d circle
        circle_3d_radius - radius of the pupil as 3d circle. Same as `diameter_3d`

        theta - circle_3d_normal described in spherical coordinates
        phi - circle_3d_normal described in spherical coordinates

        projected_sphere_center_x - x center of the 3d sphere projected back onto the eye image frame. Units are in image pixels.
        projected_sphere_center_y - y center of the 3d sphere projected back onto the eye image frame
        projected_sphere_axis_a - first axis of the 3d sphere projection.
        projected_sphere_axis_b - second axis of the 3d sphere projection.
        projected_sphere_angle - angle of the 3d sphere projection. Units are degrees.


    gaze_positions.csv
    keys:
        timestamp - timestamp of the source image frame
        index - associated_frame: closest world video frame
        confidence - computed confidence between 0 (not confident) -1 (confident)
        norm_pos_x - x position in the world image frame in normalized coordinates
        norm_pos_y - y position in the world image frame in normalized coordinates
        base_data - "timestamp-id timestamp-id ..." of pupil data that this gaze position is computed from

        #data made available by the 3d vector gaze mappers
        gaze_point_3d_x - x position of the 3d gaze point (the point the sublejct lookes at) in the world camera coordinate system
        gaze_point_3d_y - y position of the 3d gaze point
        gaze_point_3d_z - z position of the 3d gaze point
        eye_center0_3d_x - x center of eye-ball 0 in the world camera coordinate system (of camera 0 for binocular systems or any eye camera for monocular system)
        eye_center0_3d_y - y center of eye-ball 0
        eye_center0_3d_z - z center of eye-ball 0
        gaze_normal0_x - x normal of the visual axis for eye 0 in the world camera coordinate system (of eye 0 for binocular systems or any eye for monocular system). The visual axis goes through the eye ball center and the object thats looked at.
        gaze_normal0_y - y normal of the visual axis for eye 0
        gaze_normal0_z - z normal of the visual axis for eye 0
        eye_center1_3d_x - x center of eye-ball 1 in the world camera coordinate system (not avaible for monocular setups.)
        eye_center1_3d_y - y center of eye-ball 1
        eye_center1_3d_z - z center of eye-ball 1
        gaze_normal1_x - x normal of the visual axis for eye 1 in the world camera coordinate system (not avaible for monocular setups.). The visual axis goes through the eye ball center and the object thats looked at.
        gaze_normal1_y - y normal of the visual axis for eye 1
        gaze_normal1_z - z normal of the visual axis for eye 1
        '''
    def __init__(self, g_pool):
        super().__init__(g_pool)

    def init_gui(self):
        self.menu = ui.Scrolling_Menu('Raw Data Exporter')
        self.g_pool.gui.append(self.menu)

        def close():
            self.alive = False

        self.menu.append(ui.Button('Close', close))
        self.menu.append(ui.Info_Text('Export Raw Pupil Capture data into .csv files.'))
        self.menu.append(ui.Info_Text('Select your export frame range using the trim marks in the seek bar. This will affect all exporting plugins.'))
        self.menu.append(ui.Info_Text('Select your export frame range using the trim marks in the seek bar. This will affect all exporting plugins.'))
        self.menu.append(ui.Text_Input('in_mark',
                                       getter=self.g_pool.trim_marks.get_string,
                                       setter=self.g_pool.trim_marks.set_string,
                                       label='frame range to export'))
        self.menu.append(ui.Info_Text("Press the export button or type 'e' to start the export."))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def on_notify(self, notification):
        if notification['subject'] == "should_export":
            self.export_data(notification['range'], notification['export_dir'])

    def export_data(self, export_range, export_dir):
        export_range = slice(*export_range)
        with open(os.path.join(export_dir, 'pupil_positions.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')

            csv_writer.writerow(('timestamp',
                                 'index',
                                 'id',
                                 'confidence',
                                 'norm_pos_x',
                                 'norm_pos_y',
                                 'diameter',
                                 'method',
                                 'ellipse_center_x',
                                 'ellipse_center_y',
                                 'ellipse_axis_a',
                                 'ellipse_axis_b',
                                 'ellipse_angle',
                                 'diameter_3d',
                                 'model_confidence',
                                 'model_id',
                                 'sphere_center_x',
                                 'sphere_center_y',
                                 'sphere_center_z',
                                 'sphere_radius',
                                 'circle_3d_center_x',
                                 'circle_3d_center_y',
                                 'circle_3d_center_z',
                                 'circle_3d_normal_x',
                                 'circle_3d_normal_y',
                                 'circle_3d_normal_z',
                                 'circle_3d_radius',
                                 'theta',
                                 'phi',
                                 'projected_sphere_center_x',
                                 'projected_sphere_center_y',
                                 'projected_sphere_axis_a',
                                 'projected_sphere_axis_b',
                                 'projected_sphere_angle'))

            for p in list(chain(*self.g_pool.pupil_positions_by_frame[export_range])):
                data_2d = ['{}'.format(p['timestamp']),  # use str to be consitant with csv lib.
                           p['index'],
                           p['id'],
                           p['confidence'],
                           p['norm_pos'][0],
                           p['norm_pos'][1],
                           p['diameter'],
                           p['method']]
                try:
                    ellipse_data = [p['ellipse']['center'][0],
                                    p['ellipse']['center'][1],
                                    p['ellipse']['axes'][0],
                                    p['ellipse']['axes'][1],
                                    p['ellipse']['angle']]
                except KeyError:
                    ellipse_data = [None]*5
                try:
                    data_3d = [p['diameter_3d'],
                               p['model_confidence'],
                               p['model_id'],
                               p['sphere']['center'][0],
                               p['sphere']['center'][1],
                               p['sphere']['center'][2],
                               p['sphere']['radius'],
                               p['circle_3d']['center'][0],
                               p['circle_3d']['center'][1],
                               p['circle_3d']['center'][2],
                               p['circle_3d']['normal'][0],
                               p['circle_3d']['normal'][1],
                               p['circle_3d']['normal'][2],
                               p['circle_3d']['radius'],
                               p['theta'],
                               p['phi'],
                               p['projected_sphere']['center'][0],
                               p['projected_sphere']['center'][1],
                               p['projected_sphere']['axes'][0],
                               p['projected_sphere']['axes'][1],
                               p['projected_sphere']['angle']]
                except KeyError:
                    data_3d = [None]*21
                row = data_2d + ellipse_data + data_3d
                csv_writer.writerow(row)
            logger.info("Created 'pupil_positions.csv' file.")

        with open(os.path.join(export_dir, 'gaze_positions.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(("timestamp",
                                 "index",
                                 "confidence",
                                 "norm_pos_x",
                                 "norm_pos_y",
                                 "base_data",
                                 "gaze_point_3d_x",
                                 "gaze_point_3d_y",
                                 "gaze_point_3d_z",
                                 "eye_center0_3d_x",
                                 "eye_center0_3d_y",
                                 "eye_center0_3d_z",
                                 "gaze_normal0_x",
                                 "gaze_normal0_y",
                                 "gaze_normal0_z",
                                 "eye_center1_3d_x",
                                 "eye_center1_3d_y",
                                 "eye_center1_3d_z",
                                 "gaze_normal1_x",
                                 "gaze_normal1_y",
                                 "gaze_normal1_z"))

            for g in list(chain(*self.g_pool.gaze_positions_by_frame[export_range])):
                data = ['{}'.format(g["timestamp"]), g["index"], g["confidence"], g["norm_pos"][0], g["norm_pos"][1],
                        " ".join(['{}-{}'.format(b['timestamp'], b['id']) for b in g['base_data']])]  # use str on timestamp to be consitant with csv lib.

                # add 3d data if avaiblable
                if g.get('gaze_point_3d', None) is not None:
                    data_3d = [g['gaze_point_3d'][0], g['gaze_point_3d'][1], g['gaze_point_3d'][2]]

                    # binocular
                    if g.get('eye_centers_3d' ,None) is not None:
                        data_3d += g['eye_centers_3d'].get(0, [None, None, None])
                        data_3d += g['gaze_normals_3d'].get(0, [None, None, None])
                        data_3d += g['eye_centers_3d'].get(1, [None, None, None])
                        data_3d += g['gaze_normals_3d'].get(1, [None, None, None])
                    # monocular
                    elif g.get('eye_center_3d', None) is not None:
                        data_3d += g['eye_center_3d']
                        data_3d += g['gaze_normal_3d']
                        data_3d += [None]*6
                else:
                    data_3d = [None]*15
                data += data_3d
                csv_writer.writerow(data)
            logger.info("Created 'gaze_positions.csv' file.")

        with open(os.path.join(export_dir, 'pupil_gaze_positions_info.txt'), 'w', encoding='utf-8', newline='') as info_file:
            info_file.write(self.__doc__)

    def get_init_dict(self):
        return {}

    def cleanup(self):
        self.deinit_gui()
