"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
# Do not import Online/Offline_Head_Pose_Tracker plugins here!
# If disobeyed background tasks will try to import them in the background
# and fail. The plugins should not be required in the background.
