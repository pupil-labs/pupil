import numpy as np

# logging
import logging

logger = logging.getLogger(__name__)


def filter_confidence(pupil_list, threshold):

    len_pre_filter = len(pupil_list)

    pupil_list = [p for p in pupil_list if p["confidence"] >= threshold]

    len_post_filter = len(pupil_list)
    try:
        dismissed_percentage = 100 * (1.0 - len_post_filter / len_pre_filter)
    except ZeroDivisionError:
        pass  # empty pupil_list, is being handled in match_data
    else:
        logger.info(
            "Dismissing {:.2f}% pupil data due to confidence < {:.2f}".format(
                dismissed_percentage, threshold
            )
        )

    return pupil_list


def find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    try:
        if abs(value - array[idx - 1]) < abs(value - array[idx]):
            return idx - 1
        else:
            return idx
    except IndexError:
        return idx - 1


def match_data(g_pool, pupil_list, ref_list):
    """
    Returns binocular and monocular matched pupil datums and ref points. Uses a dispersion criterion to
    dismiss matches which are too far apart.
    """
    if pupil_list and ref_list:
        pass
    else:
        not_enough_data_error_msg = (
            "Not enough ref point or pupil data available for calibration."
        )
        logger.error(not_enough_data_error_msg)
        return {
            "subject": "calibration.failed",
            "reason": not_enough_data_error_msg,
            "timestamp": g_pool.get_timestamp(),
            "record": True,
        }

    pupil0 = [p for p in pupil_list if p["id"] == 0]
    pupil1 = [p for p in pupil_list if p["id"] == 1]

    matched_binocular_data = closest_matches_binocular(ref_list, pupil_list)
    matched_pupil0_data = closest_matches_monocular(ref_list, pupil0)
    matched_pupil1_data = closest_matches_monocular(ref_list, pupil1)

    if len(matched_pupil0_data) > len(matched_pupil1_data):
        matched_monocular_data = matched_pupil0_data
    else:
        matched_monocular_data = matched_pupil1_data

    logger.info(
        "Collected {} monocular calibration data.".format(len(matched_monocular_data))
    )
    logger.info(
        "Collected {} binocular calibration data.".format(len(matched_binocular_data))
    )
    return (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
        pupil0,
        pupil1,
    )


def closest_matches_binocular(ref_pts, pupil_pts, max_dispersion=1 / 15.0):
    """
    get pupil positions closest in time to ref points.
    return list of dict with matching ref, pupil0 and pupil1 data triplets.
    """
    pupil0 = [p for p in pupil_pts if p["id"] == 0]
    pupil1 = [p for p in pupil_pts if p["id"] == 1]

    pupil0_ts = np.array([p["timestamp"] for p in pupil0])
    pupil1_ts = np.array([p["timestamp"] for p in pupil1])

    matched = []

    if pupil0 and pupil1:
        for r in ref_pts:

            closest_p0_idx = find_nearest_idx(pupil0_ts, r["timestamp"])
            closest_p0 = pupil0[closest_p0_idx]
            closest_p1_idx = find_nearest_idx(pupil1_ts, r["timestamp"])
            closest_p1 = pupil1[closest_p1_idx]

            dispersion = max(
                closest_p0["timestamp"], closest_p1["timestamp"], r["timestamp"]
            ) - min(closest_p0["timestamp"], closest_p1["timestamp"], r["timestamp"])
            if dispersion < max_dispersion:
                matched.append({"ref": r, "pupil": closest_p0, "pupil1": closest_p1})
            else:
                logger.debug(
                    "Binocular match rejected due to time dispersion criterion"
                )

    return matched


def closest_matches_monocular(ref_pts, pupil_pts, max_dispersion=1 / 15.0):
    """
    get pupil positions closest in time to ref points.
    return list of dict with matching ref and pupil datum.

    if your data is binocular use:
    pupil0 = [p for p in pupil_pts if p['id']==0]
    pupil1 = [p for p in pupil_pts if p['id']==1]
    to get the desired eye and pass it as pupil_pts
    """

    pupil0 = pupil_pts
    pupil0_ts = np.array([p["timestamp"] for p in pupil0])

    matched = []

    if pupil0:
        for r in ref_pts:
            closest_p0_idx = find_nearest_idx(pupil0_ts, r["timestamp"])
            closest_p0 = pupil0[closest_p0_idx]
            dispersion = np.abs(closest_p0["timestamp"] - r["timestamp"])
            if dispersion < max_dispersion:
                matched.append({"ref": r, "pupil": closest_p0})
            else:
                logger.debug(
                    "Monocular match rejected due to time dispersion criterion"
                )

    return matched


def preprocess_2d_data_monocular(matched_data):
    """"
    Takes matched pupil data and returns list of tuples, keeping only the positions in normalized screen coordinates
    for pupil and ref.
    """
    cal_data = [
        (*pair["pupil"]["norm_pos"], *pair["ref"]["norm_pos"]) for pair in matched_data
    ]
    return cal_data


def preprocess_2d_data_binocular(matched_data):
    """
    Takes matched pupil data and returns list of tuples, keeping only the positions in normalized
    coordinates for pupil0, pupil1, and ref positions.
    """
    cal_data = [
        (
            *triplet["pupil"]["norm_pos"],
            *triplet["pupil1"]["norm_pos"],
            *triplet["ref"]["norm_pos"],
        )
        for triplet in matched_data
    ]
    return cal_data


def preprocess_3d_data(matched_data, g_pool):
    """
    Takes matched data, splits into pupil0, pupil1, ref, keeping only the normals and unprojections, respectively.
    """
    pupil0_processed = [
        dp["pupil"]["circle_3d"]["normal"]
        for dp in matched_data
        if "circle_3d" in dp["pupil"]
    ]

    pupil1_processed = [
        dp["pupil1"]["circle_3d"]["normal"]
        for dp in matched_data
        if "pupil1" in dp and "circle_3d" in dp["pupil1"]
    ]

    ref = np.array([dp["ref"]["screen_pos"] for dp in matched_data])
    ref_processed = g_pool.capture.intrinsics.unprojectPoints(ref, normalize=True)

    return ref_processed, pupil0_processed, pupil1_processed

