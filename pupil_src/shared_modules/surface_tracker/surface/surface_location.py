import numpy as np


class Surface_Location:
    def __init__(
        self,
        detected,
        dist_img_to_surf_trans=None,
        surf_to_dist_img_trans=None,
        img_to_surf_trans=None,
        surf_to_img_trans=None,
        num_detected_markers=0,
    ):
        self.detected = detected

        if self.detected:
            assert (
                dist_img_to_surf_trans is not None
                and surf_to_dist_img_trans is not None
                and img_to_surf_trans is not None
                and surf_to_img_trans is not None
                and num_detected_markers > 0
            ), (
                "Surface_Location can not be detected and have None as "
                "transformations at the same time!"
            )

        self.dist_img_to_surf_trans = dist_img_to_surf_trans
        self.surf_to_dist_img_trans = surf_to_dist_img_trans
        self.img_to_surf_trans = img_to_surf_trans
        self.surf_to_img_trans = surf_to_img_trans
        self.num_detected_markers = num_detected_markers

    def get_serializable_copy(self):
        location = {}
        location["detected"] = self.detected
        location["num_detected_markers"] = self.num_detected_markers
        if self.detected:
            location["dist_img_to_surf_trans"] = self.dist_img_to_surf_trans.tolist()
            location["surf_to_dist_img_trans"] = self.surf_to_dist_img_trans.tolist()
            location["img_to_surf_trans"] = self.img_to_surf_trans.tolist()
            location["surf_to_img_trans"] = self.surf_to_img_trans.tolist()
        else:
            location["dist_img_to_surf_trans"] = None
            location["surf_to_dist_img_trans"] = None
            location["img_to_surf_trans"] = None
            location["surf_to_img_trans"] = None
        return location

    @staticmethod
    def load_from_serializable_copy(copy):
        location = Surface_Location(detected=False)
        location.detected = copy["detected"]
        location.dist_img_to_surf_trans = np.asarray(copy["dist_img_to_surf_trans"])
        location.surf_to_dist_img_trans = np.asarray(copy["surf_to_dist_img_trans"])
        location.img_to_surf_trans = np.asarray(copy["img_to_surf_trans"])
        location.surf_to_img_trans = np.asarray(copy["surf_to_img_trans"])
        location.num_detected_markers = copy["num_detected_markers"]
        return location

    def __bool__(self):
        return self.detected
