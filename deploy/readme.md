
# Deploy Workflow
---
## Bundle using Pyinstaller
	pyinstaller -w bundle_name.spec

## Create a version file inside the distribution folder
	python write_version_string.py

## Make sure that all excecutables in `/dist/pupil` are chmodded to be exceutable
	chmod 775 pupil_capture
	chmod 775 v4l2-ctl
