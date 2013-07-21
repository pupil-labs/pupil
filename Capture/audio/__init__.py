'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License. 
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import platform
os_name = platform.system()
del platform

import subprocess as sp

# OS specific audio players via terminal
if os_name == "Linux":
	def bing():
		sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])

	def say(message):
		sp.Popen(["spd-say", message])
elif os_name == "Darwin":
	def bing():
		sp.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])

	def say(message):
		sp.Popen(["say", message])
else:
	def bing():
		print '\a'

	def say(message):
		print '\a' 
		print message
