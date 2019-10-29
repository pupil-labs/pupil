"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import argparse
import sys
import typing as T


def parse(running_from_bundle: bool, **defaults: T.Any):
    """Parse command line arguments."""
    main_parser = argparse.ArgumentParser(allow_abbrev=False)

    if running_from_bundle:
        # Infer the app from executable name and only parse those arguments
        _add_general_args(main_parser)
        if "pupil_capture" in sys.executable:
            _add_app_args(main_parser, "capture")
        elif "pupil_player" in sys.executable:
            _add_app_args(main_parser, "player")
        else:
            _add_app_args(main_parser, "service")
        main_parser.set_defaults(**defaults)

    else:
        # Add explicit subparsers for all apps
        subparser = main_parser.add_subparsers(
            dest="app", metavar="<app>", help="Application to start"
        )
        subparser.required = True

        apps = {
            "capture": "Real-time processing and recording",
            "player": "Process, visualize, and export recordings",
            "service": "Real-time processing with minimal UI",
        }

        for app, description in apps.items():
            app_parser = subparser.add_parser(app, help=description)
            _add_general_args(app_parser)
            _add_app_args(app_parser, app)
            app_parser.set_defaults(**defaults)

    return main_parser.parse_known_args()


def _add_general_args(parser: argparse.ArgumentParser):
    # Args that apply to all apps
    parser.add_argument("--version", action="store_true", help="show version")
    parser.add_argument(
        "--debug", action="store_true", help="display debug log messages"
    )
    parser.add_argument(
        "--profile", help="profile the application's CPU time", action="store_true"
    )


def _add_app_args(parser: argparse.ArgumentParser, app: str):
    # Add specific arguments based on app
    if app in ["capture", "service"]:
        parser.add_argument("-P", "--port", type=int, help="Pupil Remote port")

    if app == "player":
        parser.add_argument(
            "recording", default="", nargs="?", help="Path to Recording"
        )
