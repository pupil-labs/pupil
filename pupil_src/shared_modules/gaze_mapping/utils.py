"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _filter_pupil_list_by_confidence(pupil_list, threshold):
    if not pupil_list:
        return []

    len_pre_filter = len(pupil_list)
    pupil_list = [p for p in pupil_list if p["confidence"] >= threshold]
    len_post_filter = len(pupil_list)
    dismissed_percentage = 100 * (1.0 - len_post_filter / len_pre_filter)
    logger.debug(
        f"Dismissing {dismissed_percentage:.2f}% pupil data due to "
        f"confidence < {threshold:.2f}"
    )
    max_expected_percentage = 20.0
    if dismissed_percentage >= max_expected_percentage:
        logger.warning(
            "An unexpectedly large amount of pupil data "
            f"(> {max_expected_percentage:.0f}%) was dismissed due to low confidence. "
            "Please check the pupil detection."
        )
    return pupil_list


def _match_data_batch(pupil_list, ref_list):
    assert pupil_list, "No pupil data to match"
    assert ref_list, "No reference data to match"
    pupil0 = [p for p in pupil_list if p["id"] == 0]
    pupil1 = [p for p in pupil_list if p["id"] == 1]

    matched_binocular_data = closest_matches_binocular_batch(ref_list, pupil0, pupil1)
    matched_pupil0_data = closest_matches_monocular_batch(ref_list, pupil0)
    matched_pupil1_data = closest_matches_monocular_batch(ref_list, pupil1)

    num_bino = len(matched_binocular_data[0])
    num_mono_right = len(matched_pupil0_data[0])
    num_mono_left = len(matched_pupil1_data[0])

    logger.debug(f"Collected {num_bino} binocular references.")
    logger.debug(f"Collected {num_mono_right} right eye monocular references.")
    logger.debug(f"Collected {num_mono_left} left eye monocular references.")

    return (
        matched_binocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
    )


def closest_matches_binocular_batch(ref_pts, pupil0, pupil1, max_dispersion=1 / 15.0):
    """Get pupil positions closest in time to ref points.
    Return list of dict with matching ref, pupil0 and pupil1 data triplets.
    """

    matched = [[], [], []]
    if not (ref_pts and pupil0 and pupil1):
        return matched

    pupil0_ts = np.array([p["timestamp"] for p in pupil0])
    pupil1_ts = np.array([p["timestamp"] for p in pupil1])

    for r in ref_pts:
        closest_p0_idx = _find_nearest_idx(pupil0_ts, r["timestamp"])
        closest_p0 = pupil0[closest_p0_idx]
        closest_p1_idx = _find_nearest_idx(pupil1_ts, r["timestamp"])
        closest_p1 = pupil1[closest_p1_idx]

        dispersion = max(
            closest_p0["timestamp"], closest_p1["timestamp"], r["timestamp"]
        ) - min(closest_p0["timestamp"], closest_p1["timestamp"], r["timestamp"])
        if dispersion < max_dispersion:
            matched[0].append(r)
            matched[1].append(closest_p0)
            matched[2].append(closest_p1)
        else:
            logger.debug("Binocular match rejected due to time dispersion criterion")
    return matched


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


def closest_matches_monocular_batch(ref_pts, pupil, max_dispersion=1 / 15.0):
    """Get pupil positions closest in time to ref points.
    Return list of dict with matching ref and pupil datum.
    """

    matched = [[], []]
    if not (ref_pts and pupil):
        return matched

    pupil_ts = np.array([p["timestamp"] for p in pupil])

    for r in ref_pts:
        closest_p_idx = _find_nearest_idx(pupil_ts, r["timestamp"])
        closest_p = pupil[closest_p_idx]
        dispersion = max(closest_p["timestamp"], r["timestamp"]) - min(
            closest_p["timestamp"], r["timestamp"]
        )
        if dispersion < max_dispersion:
            matched[0].append(r)
            matched[1].append(closest_p)
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
