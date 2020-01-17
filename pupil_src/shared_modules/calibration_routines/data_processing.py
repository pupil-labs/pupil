"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging

import numpy as np

import file_methods as fm

logger = logging.getLogger(__name__)


def get_data_for_calibration(g_pool, pupil_list, ref_list, mode):
    """Returns extracted data for calibration and whether there is binocular data"""

    pupil_list = _filter_pupil_list_by_confidence(
        pupil_list, g_pool.min_calibration_confidence
    )

    matched_data = _match_data(pupil_list, ref_list)
    (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
    ) = matched_data

    binocular = None
    extracted_data = None
    if mode == "3d":
        if matched_binocular_data:
            binocular = True
            extracted_data = _extract_3d_data(g_pool, matched_binocular_data)
        elif matched_monocular_data:
            binocular = False
            extracted_data = _extract_3d_data(g_pool, matched_monocular_data)

    elif mode == "2d":
        if matched_binocular_data:
            binocular = True
            cal_pt_cloud_binocular = _extract_2d_data_binocular(matched_binocular_data)
            cal_pt_cloud0 = _extract_2d_data_monocular(matched_pupil0_data)
            cal_pt_cloud1 = _extract_2d_data_monocular(matched_pupil1_data)
            extracted_data = cal_pt_cloud_binocular, cal_pt_cloud0, cal_pt_cloud1
        elif matched_monocular_data:
            binocular = False
            cal_pt_cloud = _extract_2d_data_monocular(matched_monocular_data)
            extracted_data = (cal_pt_cloud,)

    return binocular, extracted_data


def _filter_pupil_list_by_confidence(pupil_list, threshold):
    if not pupil_list:
        return []

    len_pre_filter = len(pupil_list)
    pupil_list = [p for p in pupil_list if p["confidence"] >= threshold]
    len_post_filter = len(pupil_list)
    dismissed_percentage = 100 * (1.0 - len_post_filter / len_pre_filter)
    logger.info(
        f"Dismissing {dismissed_percentage:.2f}% pupil data due to "
        f"confidence < {threshold:.2f}"
    )
    return pupil_list


def _match_data(pupil_list, ref_list):
    """Returns binocular and monocular matched pupil datums and ref points.
    Uses a dispersion criterion to dismiss matches which are too far apart.
    """

    pupil0 = [p for p in pupil_list if p["id"] == 0]
    pupil1 = [p for p in pupil_list if p["id"] == 1]

    matched_binocular_data = closest_matches_binocular(ref_list, pupil0, pupil1)
    matched_pupil0_data = closest_matches_monocular(ref_list, pupil0)
    matched_pupil1_data = closest_matches_monocular(ref_list, pupil1)

    if len(matched_pupil0_data) > len(matched_pupil1_data):
        matched_monocular_data = matched_pupil0_data
    else:
        matched_monocular_data = matched_pupil1_data

    logger.info(f"Collected {len(matched_monocular_data)} monocular calibration data.")
    logger.info(f"Collected {len(matched_binocular_data)} binocular calibration data.")

    return (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
    )


def closest_matches_binocular(ref_pts, pupil0, pupil1, max_dispersion=1 / 15.0):
    """Get pupil positions closest in time to ref points.
    Return list of dict with matching ref, pupil0 and pupil1 data triplets.
    """

    if not (ref_pts and pupil0 and pupil1):
        return []

    pupil0_ts = np.array([p["timestamp"] for p in pupil0])
    pupil1_ts = np.array([p["timestamp"] for p in pupil1])

    matched = []
    for r in ref_pts:
        closest_p0_idx = _find_nearest_idx(pupil0_ts, r["timestamp"])
        closest_p0 = pupil0[closest_p0_idx]
        closest_p1_idx = _find_nearest_idx(pupil1_ts, r["timestamp"])
        closest_p1 = pupil1[closest_p1_idx]

        dispersion = max(
            closest_p0["timestamp"], closest_p1["timestamp"], r["timestamp"]
        ) - min(closest_p0["timestamp"], closest_p1["timestamp"], r["timestamp"])
        if dispersion < max_dispersion:
            matched.append({"ref": r, "pupil": closest_p0, "pupil1": closest_p1})
        else:
            logger.debug("Binocular match rejected due to time dispersion criterion")
    return matched


def closest_matches_monocular(ref_pts, pupil, max_dispersion=1 / 15.0):
    """Get pupil positions closest in time to ref points.
    Return list of dict with matching ref and pupil datum.
    """

    if not (ref_pts and pupil):
        return []

    pupil_ts = np.array([p["timestamp"] for p in pupil])

    matched = []
    for r in ref_pts:
        closest_p_idx = _find_nearest_idx(pupil_ts, r["timestamp"])
        closest_p = pupil[closest_p_idx]
        dispersion = max(closest_p["timestamp"], r["timestamp"]) - min(
            closest_p["timestamp"], r["timestamp"]
        )
        if dispersion < max_dispersion:
            matched.append({"ref": r, "pupil": closest_p})
    return matched


def _find_nearest_idx(array, value):
    """Find the index of the element in array which is closest to value"""

    idx = np.searchsorted(array, value, side="left")
    try:
        if abs(value - array[idx - 1]) < abs(value - array[idx]):
            return idx - 1
        else:
            return idx
    except IndexError:
        return idx - 1


def _extract_3d_data(g_pool, matched_data):
    """Takes matched data, splits into ref, pupil0, pupil1.
    Return unprojections of ref, normals of pupil0 and pupil1 and last pupils
    """

    if not matched_data:
        return None

    ref = np.array([dp["ref"]["screen_pos"] for dp in matched_data])
    ref_points_unprojected = g_pool.capture.intrinsics.unprojectPoints(
        ref, normalize=True
    )

    pupil0_normals = [
        dp["pupil"]["circle_3d"]["normal"]
        for dp in matched_data
        if "circle_3d" in dp["pupil"]
    ]
    if not pupil0_normals:
        return None

    # matched_monocular_data
    if "pupil1" not in matched_data[0]:
        last_pupil = matched_data[-1]["pupil"]
        return ref_points_unprojected, np.array(pupil0_normals), last_pupil

    # matched_binocular_data
    pupil1_normals = [
        dp["pupil1"]["circle_3d"]["normal"]
        for dp in matched_data
        if "circle_3d" in dp["pupil1"]
    ]
    if not pupil1_normals:
        return None

    last_pupil0 = matched_data[-1]["pupil"]
    last_pupil1 = matched_data[-1]["pupil1"]
    return (
        ref_points_unprojected,
        np.array(pupil0_normals),
        np.array(pupil1_normals),
        last_pupil0,
        last_pupil1,
    )


def _extract_2d_data_binocular(matched_data):
    """Takes matched pupil data and returns list of tuples, keeping only the positions
    in normalized coordinates for pupil0, pupil1 and ref positions.
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


def _extract_2d_data_monocular(matched_data):
    """Takes matched pupil data and returns list of tuples, keeping only the positions
    in normalized screen coordinates for pupil and ref.
    """

    cal_data = [
        (*pair["pupil"]["norm_pos"], *pair["ref"]["norm_pos"]) for pair in matched_data
    ]
    return cal_data


def get_data_for_calibration_hmd(pupil_list, ref_list, mode):
    """Returns extracted data for hmd calibration"""

    matched_data = _match_data_hmd(pupil_list, ref_list)
    matched_binocular_data, matched_pupil0_data, matched_pupil1_data = matched_data

    extracted_data = None
    if mode == "3d":
        fm.save_object(matched_binocular_data, "hmd_cal_data")
        extracted_data = _extract_3d_data_hmd(matched_binocular_data)

    elif mode == "2d":
        if not (matched_pupil0_data or matched_pupil1_data):
            extracted_data = None
        else:
            cal_pt_cloud0 = _extract_2d_data_monocular(matched_pupil0_data)
            cal_pt_cloud1 = _extract_2d_data_monocular(matched_pupil1_data)
            extracted_data = cal_pt_cloud0, cal_pt_cloud1

            if not cal_pt_cloud0:
                logger.warning("No matched ref<->pupil data collected for id0")
            if not cal_pt_cloud1:
                logger.warning("No matched ref<->pupil data collected for id1")

    return extracted_data


def _match_data_hmd(pupil_list, ref_list):
    """Returns binocular and monocular matched pupil datums and ref points.
    Uses a dispersion criterion to dismiss matches which are too far apart.
    """

    ref0 = [r for r in ref_list if r["id"] == 0]
    ref1 = [r for r in ref_list if r["id"] == 1]
    pupil0 = [p for p in pupil_list if p["id"] == 0]
    pupil1 = [p for p in pupil_list if p["id"] == 1]

    matched_binocular_data = closest_matches_binocular(ref_list, pupil0, pupil1)
    matched_pupil0_data = closest_matches_monocular(ref0, pupil0)
    matched_pupil1_data = closest_matches_monocular(ref1, pupil1)

    return matched_binocular_data, matched_pupil0_data, matched_pupil1_data


def _extract_3d_data_hmd(matched_data):
    """Takes matched data, splits into ref, pupil0, pupil1.
    Return mm_pos of ref, normals of pupil0 and pupil1 and last pupils
    """

    ref_points_3d = [d["ref"]["mm_pos"] for d in matched_data]
    pupil0_normals = [
        d["pupil"]["circle_3d"]["normal"]
        for d in matched_data
        if "3d" in d["pupil"]["method"]
    ]
    pupil1_normals = [
        d["pupil1"]["circle_3d"]["normal"]
        for d in matched_data
        if "3d" in d["pupil"]["method"]
    ]

    if not ref_points_3d or not pupil0_normals or not pupil1_normals:
        return None

    last_pupil0 = matched_data[-1]["pupil"]
    last_pupil1 = matched_data[-1]["pupil1"]
    return (
        np.array(ref_points_3d),
        np.array(pupil0_normals),
        np.array(pupil1_normals),
        last_pupil0,
        last_pupil1,
    )
