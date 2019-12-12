import queue
from observable import Observable
from audio_capture.model.source import PyAudioObservableDelayedDeviceSource
from audio_caputre.model.transcoder import PyAudio2PyAVTranscoder
from audio_capture.model.sink import PyAVMultipartFileSink
from pupil_audio import PyAudio2PyAVCapture


class PyAudio2PyAVObservableMultipartCapture(PyAudio2PyAVCapture, Observable):

    def __init__(
        self,
        in_name: str,
        out_path: str,
        frame_rate=None,
        channels=None,
        dtype=None,
        source_cls=None,
        transcoder_cls=None,
        sink_cls=None,
        device_monitor=None,
    ):
        self.device_name = in_name
        self.device_monitor = device_monitor
        self.shared_queue = queue.Queue()

        self.source_cls = source_cls or PyAudioObservableDelayedDeviceSource
        assert issubclass(self.source_cls, PyAudioObservableDelayedDeviceSource)

        self.transcoder_cls = transcoder_cls or PyAudio2PyAVTranscoder
        assert issubclass(self.transcoder_cls, PyAudio2PyAVTranscoder)

        self.sink_cls = sink_cls or PyAVMultipartFileSink
        assert issubclass(self.sink_cls, PyAVMultipartFileSink)

        self.transcoder = self.transcoder_cls(
            frame_rate=frame_rate, channels=channels, dtype=dtype,
        )

        self.source = self._create_delayed_source()

        self.sink = self.sink_cls(
            file_path=out_path,
            transcoder=self.transcoder,
            in_queue=self.shared_queue,
        )

    def start(self):
        self.source.start()
        # Only start source
        # Don't start sink since the source properties are not known yet

    def on_input_device_connected(self, device_info):
        # Since the source is a newly discovered device, need to update the transcoder
        self.transcoder.frame_rate = device_info.default_sample_rate
        self.transcoder.channels = device_info.max_input_channels
        if not self.sink.is_running:
            self.sink.start()

    def on_input_device_disconnected(self):

        # Re-create the source with the same parameters
        self.source.cleanup()
        self.source = self._create_delayed_source()

        # Signal to the sink to finish current part and start a new one
        self.sink.break_part()

        # Start the source; this will wait for the device to appear
        self.source.start()

    def _create_delayed_source(self):
        source = self.source_cls(
            device_index=None,
            device_name=self.device_name,
            device_monitor=self.device_monitor,
            frame_rate=self.transcoder.frame_rate,
            channels=self.transcoder.channels,
            format=self.transcoder.pyaudio_format,
            out_queue=self.shared_queue,
        )
        source.add_observer("on_input_device_connected", self.on_input_device_connected)
        source.add_observer("on_input_device_disconnected", self.on_input_device_disconnected)
        return source
