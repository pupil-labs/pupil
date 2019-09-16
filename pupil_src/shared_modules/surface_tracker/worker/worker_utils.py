import platform
import multiprocessing


# On macOS, "spawn" is set as default start method in main.py. This is not required
# here and we set it back to "fork" to improve performance.
if platform.system() == "Darwin":
    mp_context = multiprocessing.get_context("fork")
else:
    mp_context = multiprocessing.get_context()
