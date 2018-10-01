"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import logging
import os
import uuid

import file_methods as fm


logger = logging.getLogger(__name__)


class TokenManger:
    __file_version__ = 1

    def __init__(self, name, inputs=(), outputs=()):
        self.name = name
        self.inputs = {key: None for key in inputs}
        self.outputs = {key: None for key in outputs}

    def on_input_token(self, input_, token):
        """Should be called by client if new input tokens are available.

        This function decides if action is required depending on whether the token
        has been encountered before.
        """
        if self.inputs[input_] == token:
            logger.debug("Token for input '{}' already received".format(input_))
            return
        self.act_on_unseen_input_token(input_)

    def act_on_unseen_input_token(self, input_=None):
        """Called from on_input_token if action is required, should be overwritten or
        replaced by client with actual action function.

        acted_on_input_token() should be called after the action has been finished.

        Tip: acted_on_input_token() can be called asynchronously
        """
        self.acted_on_input_token(input_)

    def acted_on_input_token(self, input_=None):
        """called when finished acting on input"""
        for output_ in self.outputs:
            token = self.generate_token()
            self.act_on_new_output_token(output_, token)
            self.outputs[output_] = token

    @abc.abstractmethod
    def act_on_new_output_token(self, output_, token):
        """e.g. announce change, save filtered data"""
        return NotImplemented

    @abc.abstractmethod
    def generate_token(self):
        raise NotImplementedError

    def update_from_file(self, directory):
        try:
            loaded = fm.load_object(self._file_path(directory))
            assert loaded["version"] == self.__file_version__
        except AssertionError:
            logger.debug("Deprecated token format. Not updating tokens.")
        except FileNotFoundError:
            logger.debug("No persistent tokens found. Not updating tokens.")
        else:
            self.inputs.update(loaded["inputs"])
            self.outputs.update(loaded["outputs"])

    def save_to_file(self, directory):
        fm.save_object(
            {
                "inputs": self.inputs,
                "outputs": self.outputs,
                "version": self.__file_version__,
            },
            self._file_path(directory),
        )

    def _file_path(self, directory):
        return os.path.join(directory, self.name + ".tokens")


class UUIDTokenManager(TokenManger):
    def generate_token(self):
        return uuid.uuid4()


class ConstantTokenManager(TokenManger):
    def generate_token(self):
        return self.name
