'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import cPickle as pickle
import os 
import json
import logging
logger = logging.getLogger(__name__)

class Persistent_Dict(dict):
	"""a dict class that uses json to save itself to file. backward-compatible with pickled objects."""
	def __init__(self, file_path):
		super(Persistent_Dict, self).__init__()
		self.file_path = os.path.expanduser(file_path)

		# pickled files
		if os.path.exists(self.file_path):
			with open(self.file_path,'rb') as fh:
				self.update(pickle.load(fh))
		elif os.path.exists(self.file_path+'.json'):
			with open(self.file_path+'.json') as fh:
				self.update(json.load(fh))
		else:
			logger.debug("Session settings file '{}' not found. Will make new one on exit.".format(self.file_path))
		

	def save(self):
		logger.debug("Saving {}".format(self.file_path))
		try:
			with open(self.file_path+'.json','w') as fh:
				json.dump(self, fh, sort_keys=True, indent=2)
			# remove old pickled file
			if os.path.exists(self.file_path):
				logger.debug("Removing old '{}' pickled file.".format(self.file_path))
				os.remove(self.file_path)

		except IOError:
			logger.warning("Could not save session settings to '%s'"%self.file_path)

	

	def close(self):
		self.save()

	
def save_object(object,file_path):
	file_path = os.path.expanduser(file_path)
	with open(file_path, 'w') as fh:
		json.dump(self, fh, sort_keys=True, indent=2)

if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	settings = Persistent_Dict('~/Desktop/pupil_settings/user_settings_eye')
	print settings['roi']