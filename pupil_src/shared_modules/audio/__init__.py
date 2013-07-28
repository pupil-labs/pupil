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
from time import sleep

import subprocess as sp

# OS specific audio players via terminal
if os_name == "Linux":
	def beep():
		sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])

	def tink():
		sp.Popen(["paplay", "/usr/share/sounds/ubuntu/stereo/message.ogg"])

	def say(message):
		sp.Popen(["spd-say", message])

elif os_name == "Darwin":
	def beep():
		sp.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])

	def tink():
		sp.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])

	def say(message):
		sp.Popen(["say", message, "-v" "Victoria"])
else:
	def beep():
		print '\a'

	def tink():
		print '\a'

	def say(message):
		print '\a'
		print message

if __name__ == '__main__':
	beep()
	sleep(1)
	tink()
	sleep(1)
	say("Hello, I am Pupil's audio module.")