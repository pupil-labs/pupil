
# Deploy Workflow
---
## Bundle using Pyinstaller
	pyinstaller -w bundle_**os_name**.spec

## Create a version file inside the distribution folder,chmod stuff...
	python finalize_bundle_**os_name**.py


pre-requisits:
brew python (not! system python)
pip install pyinstaller

