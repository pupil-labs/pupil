
class Plugin(object):
    """docstring for Plugin

    plugin is a base class
    it has all interfaces that will be called
    instances of this class ususally get added to a plugins list
    this list will have its members called with all methods invoked.

    Creating an ATB Bar in __init__ is required in the class that is based on this class
    Show at least some info about the Ref_Detector
    self._bar = atb.Bar(name = self.__class__.__name__, label='your_label',
                help="ref detection parameters", color=(50, 50, 50), alpha=100,
                text='light', position=atb_pos,refresh=.3, size=(300, 150))
    """
    def __init__(self):
        self._alive = True

    @property
    def alive(self):
        """This field indicates of the instance should be detroyed
        Writing False to this will schedule the instance for deletion
        """
        if not self._alive:
            if hasattr(self,"_bar"):
                try:
                    self._bar.destroy()
                    del self._bar
                except:
                    print "Tried to delete an already dead bar. This is a bug. Please report"
        return self._alive

    @alive.setter
    def alive(self, value):
        if isinstance(value,bool):
            self._alive = value

    def on_click(self,pos):
        """
        gets called when the user clicks in the window screen
        """
        pass

    def update(self,img):
        """
        gets called once every frame
        """
        pass

    def gl_display(self):
        """
        gets called once every frame
        """
        pass

    def __del__(self):
        pass


class Ref_Detector_Template(Plugin):
    """
    template of reference detectors class
    build a detector with this as your template.

    Your derived class needs to have interfaces
    defined by these methods:
    you NEED to do at least what is done in these fn-prototypes

    """
    def __init__(self, global_calibrate, shared_pos, screen_marker_pos, screen_marker_state, atb_pos=(0,0)):
        Plugin.__init__(self)

        self.active = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_pos = shared_pos
        self.shared_screen_marker_pos = screen_marker_pos
        self.shared_screen_marker_state = screen_marker_state
        self.screen_marker_state = -1
        # indicated that no pos has been found
        self.shared_pos = 0,0


        # Creating an ATB Bar required Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "A_Unique_Name", label="",
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 150))
        self._bar.add_button("  begin calibrating  ", self.start)

    def start(self):
        self.global_calibrate.value = True
        self.shared_pos[:] = 0,0
        self.active = True

    def stop(self):
        self.global_calibrate.value = False
        self.shared_pos[:] = 0,0
        self.screen_marker_state = -1
        self.active = False


    def update(self,img):
        if self.active:
            pass
        else:
            pass

    def __del__(self):
        self.stop()

