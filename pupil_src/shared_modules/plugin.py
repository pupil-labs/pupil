
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

    def update(self,img,recent_pupil_posotions):
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

