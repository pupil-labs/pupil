"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import argparse
import sys
import typing as T
from gettext import gettext as _


class HelpfulArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints the full help message on error."""

    def error(self, message: str):
        # NOTE: This is mostly argparse source code with slight adjustments
        args = {"prog": self.prog, "message": message}
        self._print_message(_("%(prog)s: error: %(message)s\n") % args, sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)


class PupilArgParser:
    def parse(self, running_from_bundle: bool, **defaults: T.Any):
        """Parse command line arguments for Pupil apps."""
        self.apps = {
            "capture": "real-time processing and recording",
            "player": "process, visualize, and export recordings",
            "service": "low latency real-time processing with constrained feature set",
        }

        self.main_parser = HelpfulArgumentParser(allow_abbrev=False)

        if running_from_bundle:
            self._init_bundle_parser(**defaults)
        else:
            self._init_source_parser(**defaults)

        return self.main_parser.parse_known_args()

    def _init_bundle_parser(self, **defaults):
        for app, description in self.apps.items():
            if f"pupil_{app}" in sys.executable:
                # Use main parser as a single parser for this app
                self._add_general_args(self.main_parser)
                self._add_app_args(self.main_parser, app)
                self.main_parser.description = description
                self.main_parser.set_defaults(app=app, **defaults)
                break
        else:
            raise RuntimeError(
                f"Could not infer Pupil App from executable name: {sys.executable}"
            )

    def _init_source_parser(self, **defaults):
        # Add explicit subparsers for all apps to main parser
        subparser = self.main_parser.add_subparsers(
            dest="app", metavar="<app>", help="which application to start"
        )
        subparser.required = True

        for app, description in self.apps.items():
            app_parser = subparser.add_parser(app, help=description)
            self._add_general_args(app_parser)
            self._add_app_args(app_parser, app)
            app_parser.set_defaults(**defaults)

    def _add_general_args(self, parser: argparse.ArgumentParser):
        # Args that apply to all apps
        parser.add_argument("--version", action="store_true", help="show version")
        parser.add_argument(
            "--debug", action="store_true", help="display debug log messages"
        )
        parser.add_argument(
            "--profile", action="store_true", help="profile the application's CPU time"
        )

    def _add_app_args(self, parser: argparse.ArgumentParser, app: str):
        # Args that are app specific
        if app in ["capture", "service"]:
            parser.add_argument("-P", "--port", type=int, help="port for Pupil Remote")
            parser.add_argument(
                "--hide-ui", action="store_true", help="hide ui on startup"
            )
            parser.add_argument(
                "-skip-driv",
                "--skip-driver-installation",
                action="store_true",
                help="skip automatic driver installation on Windows",
            )

        if app == "player":
            parser.add_argument(
                "recording", default="", nargs="?", help="path to recording"
            )
