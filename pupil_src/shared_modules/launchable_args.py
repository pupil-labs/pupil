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
        if "pupil_capture" in sys.executable:
            app = "capture"
        elif "pupil_player" in sys.executable:
            app = "player"
        elif "pupil_service" in sys.executable:
            app = "service"
        else:
            raise RuntimeError(
                f"Could not infer Pupil App from executable name: {sys.executable}"
            )
        _add_general_args(main_parser)
        _add_app_args(main_parser, app)
        main_parser.set_defaults(app=app, **defaults)

    else:
        # Add explicit subparsers for all apps
        subparser = main_parser.add_subparsers(
            dest="app", metavar="<app>", help="which application to start"
        )
        subparser.required = True

        apps = {
            "capture": "real-time processing and recording",
            "player": "process, visualize, and export recordings",
            "service": "real-time processing with minimal UI",
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
        "--profile", action="store_true", help="profile the application's CPU time"
    )
    parser.add_argument("--hideui", action="store_true", help="hide ui on startup")


def _add_app_args(parser: argparse.ArgumentParser, app: str):
    # Add specific arguments based on app
    if app in ["capture", "service"]:
        parser.add_argument("-P", "--port", type=int, help="port for Pupil Remote")

    if app == "player":
        parser.add_argument(
            "recording", default="", nargs="?", help="path to recording"
        )
