# -*- coding: utf-8 -*-
# Author: Douglas Creager <dcreager@dcreager.net>
# This file is placed into the public domain.

# Calculates the current version number.  If possible, this is the
# output of “git describe”, modified to conform to the versioning
# scheme that setuptools uses.  If “git describe” returns an error
# (most likely because we're in an unpacked copy of a release tarball,
# rather than in a git working copy), then we fall back on reading the
# contents of the RELEASE-VERSION file.
#
# To use this script, simply import it your setup.py file, and use the
# results of get_git_version() as your package version:
#
# from version import *
#
# setup(
#     version=get_git_version(),
#     .
#     .
#     .
# )
#
# This will automatically update the RELEASE-VERSION file, if
# necessary.  Note that the RELEASE-VERSION file should *not* be
# checked into git; please add it to your top-level .gitignore file.
#
# You'll probably want to distribute the RELEASE-VERSION file in your
# sdist tarballs; to do this, just create a MANIFEST.in file that
# contains the following line:
#
#   include RELEASE-VERSION

from subprocess import Popen, PIPE,CalledProcessError
import sys, os
import subprocess

import logging
logger = logging.getLogger(__name__)

def get_tag_commit():
    """
    returns string: 'tag'-'commits since tag'-'7 digit commit id'
    """
    try:
        return subprocess.check_output(['git', 'describe'],stderr=subprocess.STDOUT,cwd=os.path.dirname(os.path.abspath(__file__)))
    except CalledProcessError as e:
        logger.error('Error calling git: "%s" output: "%s"'%(e,e.output))
        return None
    except OSError as e:
        logger.error('Could not call git, is it installed? error msg: "%s"'%e)
        return None

def dpkg_deb_version():
    '''
    [major].[minor].[rev]-[trailing-untagged-commits]
    '''
    version = get_tag_commit()
    if version is None:
        raise ValueError('Version Error')
    if '-' in version:
        parts = version.split('-')
        parts[-2] = '-'+parts[-2]
        version = '.'.join(parts[:-2])
        version = version[1:]+parts[-2]
        return version
    else:
        return version[1:]

def pupil_version():
    '''
    [major].[minor].[rev].[trailing-untagged-commits]
    '''
    version = get_tag_commit()
    if version is None:
        raise ValueError('Version Error')
    if '-' in version:
        parts = version.split('-')
        version = '.'.join(parts[:-1])
        version = version[1:]
        return version
    else:
        return version[1:]





def write_version_file(target_dir):
    version = pupil_version()
    print "Current version of Pupil: ",version

    with open(os.path.join(target_dir,'_version_string_'),'w') as f:
        f.write(version)
    print 'Wrote version into: %s' %os.path.join(target_dir,'_version_string_')

if __name__ == "__main__":
    print get_tag_commit()
