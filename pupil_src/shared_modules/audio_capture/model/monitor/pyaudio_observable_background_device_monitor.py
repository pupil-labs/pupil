from observable import Observable
from pupil_audio import DeviceInfo
from pupil_audio import PyAudioBackgroundDeviceMonitor


class PyAudioObservableBackgroundDeviceMonitor(PyAudioBackgroundDeviceMonitor, Observable):

    @property
    def devices_by_name(self) -> T.Mapping[str, DeviceInfo]:
        return PyAudioBackgroundDeviceMonitor.devices_by_name.fget(self)

    @devices_by_name.setter
    def devices_by_name(self, new_devices_by_name: T.Mapping[str, DeviceInfo]):
        old_devices_by_name = self.devices_by_name

        old_names = set(old_devices_by_name.keys())
        new_names = set(new_devices_by_name.keys())

        connected_names = new_names.difference(old_names)
        existing_names = new_names.intersection(old_names)
        disconnected_names = old_names.difference(new_names)

        for name in connected_names:
            device_info = new_devices_by_name[name]
            old_devices_by_name[name] = device_info
            self.on_device_connected(device_info)

        for name in existing_names:
            device_info = new_devices_by_name[name]
            if device_info != old_devices_by_name[name]:
                old_devices_by_name[name] = device_info
                self.on_device_updated(device_info)

        for name in disconnected_names:
            device_info = old_devices_by_name[name]
            del old_devices_by_name[name]
            self.on_device_disconnected(device_info)

        PyAudioBackgroundDeviceMonitor.devices_by_name.fset(self, old_devices_by_name)

    def on_device_connected(self, device_info):
        pass

    def on_device_updated(self, device_info):
        pass

    def on_device_disconnected(self, device_info):
        pass
