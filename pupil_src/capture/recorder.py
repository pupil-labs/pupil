from plugin import Plugin

class Recorder(Plugin):
	"""Capture Recorder"""
	def __init__(self, arg):
		super(Recorder, self).__init__()
		self.arg = arg
