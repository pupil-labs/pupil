'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cPickle as pickle
import os
import logging
logger = logging.getLogger(__name__)

class Persistent_Dict(dict):
	"""a dict class that uses pickle to save inself to file"""
	def __init__(self, file_path):
		super(Persistent_Dict, self).__init__()
		self.file_path = os.path.expanduser(file_path)
		try:
			with open(self.file_path,'rb') as fh:
				try:
					self.update(pickle.load(fh))
				except: #KeyError,EOFError
					logger.warning("Session settings file '%s'could not be read. Will overwrite on exit."%self.file_path)
		except IOError:
			logger.debug("Session settings file '%s' not found. Will make new one on exit."%self.file_path)


	def save(self):
		d = {}
		d.update(self)
		try:
			with open(self.file_path,'wb') as fh:
				pickle.dump(d,fh,-1)
		except IOError:
			logger.warning("Could not save session settings to '%s'"%self.file_path)


	def close(self):
		self.save()


def load_object(file_path):
	file_path = os.path.expanduser(file_path)
	#reading to string and loads is 2.5x faster that using the file handle and load.
	with open(file_path,'rb') as fh:
		data = fh.read()
	return pickle.loads(data)

def save_object(object,file_path):
	file_path = os.path.expanduser(file_path)
	data = pickle.dumps(object,-1)
	with open(file_path,'wb') as fh:
		data = fh.write(data)


if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	# settings = Persistent_Dict("~/Desktop/test")
	# settings['f'] = "this is a test"
	# settings['list'] = ["list 1","list2"]
	# settings.close()

	# save_object("string",'test')
	# print load_object('test')
	# settings = Persistent_Dict('~/Desktop/pupil_settings/user_settings_eye')
	# print settings['roi']
	import time
	t = time.time()
	l = load_object('/Users/mkassner/Downloads/wrp_berlin_outside_sunny_binocular/pupil_data')

