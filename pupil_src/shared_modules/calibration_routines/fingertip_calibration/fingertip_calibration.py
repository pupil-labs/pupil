'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
import audio
from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_polyline, draw_progress, RGBA
from calibration_routines.calibration_plugin_base import Calibration_Plugin
from calibration_routines.finish_calibration import finish_calibration
import torch
from calibration_routines.fingertip_calibration.models.unet import UNet
from calibration_routines.fingertip_calibration.models.ssd_lite import build_ssd_lite

# logging
import logging
logger = logging.getLogger(__name__)


class Fingertip_Calibration(Calibration_Plugin):
    """Calibrate gaze parameters using your fingertip.
       Move your head for example horizontally and vertically while gazing at your fingertip
       to quickly sample a wide range gaze angles.
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.menu = None

        ### Initialize CNN pipeline ###
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        module_folder = os.path.join(os.path.split(__file__)[0], "weights")

        # Hand Detector
        self.hand_detector_cfg = {
            'input_size': 225,
            'max_num_detection': 1,
            'nms_thresh': 0.45,
            'conf_thresh': 0.8,
            'resume_path': os.path.join(module_folder, 'hand_detector_model.pkl'),
        }
        self.hand_transform = BaseTransform(self.hand_detector_cfg['input_size'], (117.77, 115.42, 107.29), (72.03, 69.83, 71.43))
        self.hand_detector = build_ssd_lite(self.hand_detector_cfg)
        self.hand_detector.load_state_dict(torch.load(self.hand_detector_cfg['resume_path']))
        self.hand_detector.eval().to(self.device)

        # Fingertip Detector
        self.fingertip_detector_cfg = {
            'conf_thresh': 0.7,
            'resume_path': os.path.join(module_folder, "fingertip_detector_model.pkl"),
        }
        self.fingertip_transform = BaseTransform(64, (121.97, 119.65, 111.42), (67.58, 65.17, 67.72))
        self.fingertip_detector = UNet(num_classes=10, in_channels=3, depth=4, start_filts=32, up_mode='transpose')
        self.fingertip_detector.load_state_dict(torch.load(self.fingertip_detector_cfg['resume_path']))
        self.fingertip_detector.eval().to(self.device)

        self.collect_tips = False
        self.visualize = True
        self.hand_viz = []
        self.finger_viz = []

    def init_ui(self):
        super().init_ui()
        self.menu.label = 'Fingertip Calibration'
        self.menu.append(ui.Info_Text('Calibrate gaze parameters using your fingertip!'))
        self.menu.append(ui.Info_Text('Hold your index finger still at the center of the field of view of the world camera. '
                                      'Move your head horizontally and then vertically while gazing at your fingertip.'
                                      'Then show five fingers to finish the calibration.'))
        if self.device == torch.device("cpu"):
            self.menu.append(ui.Info_Text('* No GPU utilized for fingertip detection network. '
                                          'Note that the frame rate will drop during fingertip detection.'))
        else:
            self.menu.append(ui.Info_Text('* GPUs utilized for fingertip detection network'))

        self.vis_toggle = ui.Thumb('visualize', self, setter=self.toggle_vis, label='V', hotkey='v')
        self.g_pool.quickbar.append(self.vis_toggle)

    def start(self):
        if not self.g_pool.capture.online:
            logger.error("This calibration requires world capture video input.")
            return
        super().start()
        audio.say("Starting Fingertip Calibration")
        logger.info("Starting Fingertip Calibration")

        self.active = True
        self.ref_list = []
        self.pupil_list = []

    def stop(self):
        # TODO: redundancy between all gaze mappers -> might be moved to parent class
        audio.say("Stopping Fingertip Calibration")
        logger.info('Stopping Fingertip Calibration')
        self.active = False
        self.button.status_text = ''
        if self.mode == 'calibration':
            finish_calibration(self.g_pool, self.pupil_list, self.ref_list)
        elif self.mode == 'accuracy_test':
            self.finish_accuracy_test(self.pupil_list, self.ref_list)
        super().stop()

    def recent_events(self, events):
        frame = events.get('frame')
        if (self.visualize or self.active) and frame:
            orig_img = frame.img
            img_width = frame.width
            img_height = frame.height
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            # Hand Detection
            x = orig_img.copy()
            x = self.hand_transform(x)
            x = x.to(self.device)
            hand_detections = self.hand_detector(x).detach().numpy()[0][1]

            self.hand_viz = []
            self.finger_viz = []
            for hand_detection in hand_detections:
                conf, x1, y1, x2, y2 = hand_detection
                if conf == 0:
                    break

                x1 *= img_width
                x2 *= img_width
                y1 *= img_height
                y2 *= img_height
                self.hand_viz.append((x1, y1, x2, y2))

                tl = np.array((x1, y1))
                br = np.array((x2, y2))
                W, H = br - tl
                crop_len = np.clip(max(W, H) * 1.5, 1, min(img_width, img_height))
                crop_center = (br + tl) / 2
                crop_center = np.clip(crop_center, crop_len / 2, (img_width, img_height) - crop_len / 2)
                crop_tl = (crop_center - crop_len / 2).astype(np.int)
                crop_br = (crop_tl + crop_len).astype(np.int)

                # Fingertip detection
                y = orig_img[crop_tl[1]:crop_br[1], crop_tl[0]:crop_br[0]].copy()
                y = self.fingertip_transform(y)
                y = y.to(self.device)
                fingertip_detections = self.fingertip_detector(y).cpu().detach().numpy()[0][:5]

                self.finger_viz.append([])
                detected_fingers = 0
                ref = None
                for fingertip_detection in fingertip_detections:
                    p = np.unravel_index(fingertip_detection.argmax(), fingertip_detection.shape)
                    if fingertip_detection[p] >= self.fingertip_detector_cfg['conf_thresh']:
                        p = np.array(p) / (fingertip_detection.shape) * (crop_br[1] - crop_tl[1], crop_br[0] - crop_tl[0]) + (crop_tl[1], crop_tl[0])
                        self.finger_viz[-1].append(p)
                        detected_fingers += 1
                        ref = p
                    else:
                        self.finger_viz[-1].append(None)

                if detected_fingers == 1 and self.active:
                    y, x = ref
                    ref = {
                        'screen_pos': (x, y),
                        'norm_pos': (x / img_width, 1 - (y / img_height)),
                        'timestamp': frame.timestamp,
                    }
                    self.ref_list.append(ref)
                elif detected_fingers == 5 and self.active:
                    if self.collect_tips and len(self.ref_list) > 5:
                        self.collect_tips = False
                        self.stop()
                    elif not self.collect_tips:
                        self.collect_tips = True
                elif detected_fingers == 0:
                    # hand detections without fingertips are false positives
                    del self.hand_viz[-1]
                    del self.finger_viz[-1]

            if self.active:
                # always save pupil positions
                self.pupil_list.extend(events['pupil_positions'])

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """
        if self.active or self.visualize:
            # Draw hand detection results
            for (x1, y1, x2, y2), fingertips in zip(self.hand_viz, self.finger_viz):
                pts = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]], np.int32)
                draw_polyline(pts, thickness=3 * self.g_pool.gui_user_scale, color=RGBA(0., 1., 0., 1.))
                for p in fingertips:
                    if p is not None:
                        p_flipped = p[::-1]
                        draw_progress(p_flipped, 0., 1.,
                                      inner_radius=25 * self.g_pool.gui_user_scale,
                                      outer_radius=35 * self.g_pool.gui_user_scale,
                                      color=RGBA(1., 1., 1., 1.),
                                      sharpness=0.9)

                        draw_points([p_flipped], size=10 * self.g_pool.gui_user_scale,
                                    color=RGBA(1., 1., 1., 1.),
                                    sharpness=0.9)

    def deinit_ui(self):
        """gets called when the plugin get terminated.
           either voluntarily or forced.
        """
        if self.active:
            self.stop()
        self.g_pool.quickbar.remove(self.vis_toggle)
        self.vis_toggle = None
        super().deinit_ui()

    def toggle_vis(self, _=None):
        self.visualize = not self.visualize


class BaseTransform:
    def __init__(self, size=None, mean=None, std=None):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, image):
        x = image.astype(np.float32)
        if self.size is not None:
            x = cv2.resize(x, dsize=(self.size, self.size))
        if self.mean is not None:
            x -= self.mean
        if self.std is not None:
            x /= self.std
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.unsqueeze(0)
        return x