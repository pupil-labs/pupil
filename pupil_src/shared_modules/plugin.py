'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import sys
import importlib
from time import time
import logging
logger = logging.getLogger(__name__)
'''
A simple example Plugin: 'display_recent_gaze.py'
It is a good starting point to build your own plugin.
'''


class Plugin(object):
    """docstring for Plugin
    plugin is a base class
    it has all interfaces that will be called
    instances of this class usually get added to a plugins list
    this list will have its members called with all methods invoked.

    """
    # if you have a plugin that can exist multiple times make this false in your derived class
    uniqueness = 'by_class'
    # uniqueness = 'not_unique'
    # uniqueness = 'by_base_class'

    # between 0 and 1 this indicated where in the plugin excecution order you plugin lives:
    # <.5  are things that add/mofify information that will be used by other plugins and rely on untouched data.
    # You should not edit frame if you are here!
    # == .5 is the default.
    # >.5 are things that depend on other plugins work like display , saving and streaming
    # you can change this in __init__ for your instance or in the class definition
    order = .5
    alive = True

    # menu icon font, possible values `roboto`, `opensans`, `pupil_icons`,
    # or custom loaded font name
    icon_font = 'roboto'
    icon_chr = '?'  # character shown in menu icon

    def __init__(self, g_pool):
        self.g_pool = g_pool

    def init_ui(self):
        '''
        Called when the context will have a gl window with us. You can do your init for that here.
        '''
        pass

    def recent_events(self, events):
        '''
        Called in Player and Capture.
        Gets called once every frame.
        If you plan to update data inplace, note that this will affect all plugins executed after you.
        Use self.order to deal with this appropriately
        '''
        pass

    def gl_display(self):
        """
        Gets called once every frame when its time to draw onto the gl canvas.
        """
        pass

    def on_click(self, pos, button, action):
        """
        Gets called when the user clicks in the window screen and the event has
        not been consumed by the GUI.
        """
        pass

    def on_pos(self, pos):
        """
        Gets called when the user moves the mouse in the window screen.
        """
        pass

    def on_key(self, key, scancode, action, mods):
        """
        Gets called on key events that were not consumed by the GUI.

        See http://www.glfw.org/docs/latest/input_guide.html#input_key for
        more information key events.
        """
        pass

    def on_char(self, character):
        """
        Gets called on char events that were not consumed by the GUI.

        See http://www.glfw.org/docs/latest/input_guide.html#input_char for
        more information char events.
        """
        pass

    def on_drop(self, paths):
        """
        Gets called on dropped paths of files and/or directories on the window.

        See http://www.glfw.org/docs/latest/input_guide.html#path_drop for
        more information.
        """
        pass

    def on_window_resize(self, window, w, h):
        '''
        gets called when user resizes window.
        window is the glfw window handle of the resized window.
        '''
        pass

    def on_notify(self, notification):
        """
        this gets called when a plugin wants to notify all others.
        notification is a dict in the format {'subject':'notification_category.notification_name',['addional_field':'blah']}
        implement this fn if you want to deal with notifications
        note that notifications are collected from all threads and processes and dispatched in the update loop.
        this callback happens in the main thread.
        """
        pass

    # if you want a session persistent plugin implement this function:
    def get_init_dict(self):
        # aise NotImplementedError() if you dont want you plugin to be persistent.

        d = {}
        # add all aguments of your plugin init fn with paramter names as name field
        # do not include g_pool here.
        return d

    def deinit_ui(self):
        '''
        Called when the context will have a ui with window. You can do your deinit for that here.
        '''
        pass

    def cleanup(self):
        """
        gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have an gui or glfw window destroy it here.
        """
        pass

    # ------- do not change methods, properties below this line in your derived class

    def notify_all(self, notification):
        """

        Do not overwrite this method.

        Call `notify_all` to notify all other plugins and processes with a notification:

        notification is a dict in the format {'subject': 'notification_category.[subcategory].action_name',
        ['addional_field':'foo']}

            adding 'timestamp':self.g_pool.get_timestamp() will allow other plugins
            to know when you created this notification.

            adding 'record':True will make recorder save the notification during recording.

            adding 'remote_notify':'all' will send the event all other pupil sync nodes in the same group.
            (Remote notifyifactions are not be recevied by any local actor.)

            adding 'remote_notify':node_UUID will send the event the pupil sync nodes with node_UUID.
            (Remote notifyifactions are not be recevied by any local actor.)

            adding 'delay':3.2 will delay the notification for 3.2s.
            If a new delayed notification of same subject is sent before 3.2s
            have passed we will discard the former notification.

        You may add more fields as you like.

        All notifications must be serializable by msgpack.

        """
        if self.g_pool.app == 'exporter':
            if notification.get('delay', 0):
                notification['_notify_time_'] = time()+notification['delay']
                self.g_pool.delayed_notifications[notification['subject']] = notification
            else:
                self.g_pool.notifications.append(notification)
        else:
            self.g_pool.ipc_pub.notify(notification)

    @property
    def this_class(self):
        '''
        this instance's class
        '''
        return self.__class__

    @property
    def class_name(self):
        '''
        name of this instance's class
        '''
        return self.__class__.__name__

    @property
    def base_class(self):
        '''
        rightmost base class of this instance's class
        this way you can inherit from muliple classes and use the rightmost as your plugin group classifier
        '''
        return self.__class__.__bases__[-1]

    @property
    def base_class_name(self):
        '''
        base class name of this instance's class
        '''
        return self.base_class.__name__

    @property
    def pretty_class_name(self):
        return self.class_name.replace('_', ' ')

    def add_menu(self):
        '''
        This fn is called when the plugin ui is initialized. Do not change!
        '''
        from pyglui import ui

        def toggle_menu(collapsed):
            # This is the menu toggle logic.
            # Only one menu can be open.
            # If no menu is open the menu_bar should collapse.
            self.g_pool.menubar.collapsed = collapsed
            for m in self.g_pool.menubar.elements:
                m.collapsed = True
            self.menu.collapsed = collapsed

        def close():
            toggle_menu(True)
            self.alive = False

        # Here we make a menu and icon
        self.menu = ui.Growing_Menu('Unnamed Menu', header_pos='headline')
        if self.uniqueness == 'not_unique':
            self.menu.append(ui.Button('Close', close))
        self.menu_icon = ui.Icon('collapsed', self.menu, label=self.icon_chr,
                                 label_font=self.icon_font, on_val=False, off_val=True,
                                 setter=toggle_menu)
        self.menu_icon.order = 0.5
        self.menu_icon.tooltip = self.pretty_class_name
        self.g_pool.menubar.append(self.menu)
        self.g_pool.iconbar.append(self.menu_icon)
        toggle_menu(False)

    def remove_menu(self):
        if not self.menu.collapsed:
            self.g_pool.menubar.collapsed = True
        self.g_pool.menubar.remove(self.menu)
        self.g_pool.iconbar.remove(self.menu_icon)
        self.menu = None
        self.menu_icon = None


# Plugin manager classes and fns
class Plugin_List(object):
    """This is the Plugin Manager
        It is a self sorting list with a few functions to manage adding and
        removing Plugins and lacking most other list methods.
    """
    def __init__(self, g_pool, plugin_initializers):
        self._plugins = []
        self.g_pool = g_pool
        plugin_by_name = g_pool.plugin_by_name

        # add self as g_pool.plguins object to allow plugins to call the plugins list during init.
        # this will be done again when the init returns but is kept there for readablitly.
        self.g_pool.plugins = self

        # now add plugins to plugin list.
        for initializer in plugin_initializers:
            name, args = initializer
            logger.debug("Loading plugin: {} with settings {}".format(name, args))
            try:
                plugin_by_name[name]
            except KeyError:
                logger.debug("Plugin '{}' failed to load. Not available for import." .format(name))
            else:
                self.add(plugin_by_name[name], args)

    def __iter__(self):
        for p in self._plugins:
            yield p

    def __str__(self):
        return 'Plugin List: {}'.format(self._plugins)

    def add(self, new_plugin, args={}):
        '''
        add a plugin instance to the list.
        '''
        if new_plugin.uniqueness == 'by_base_class':
            for p in self._plugins:
                if p.base_class == new_plugin.__bases__[-1]:
                    replc_str = "Plugin {} of base class {} will be replaced by {}."
                    logger.debug(replc_str.format(p, p.base_class_name, new_plugin.__name__))
                    p.alive = False
                    self.clean()

        elif new_plugin.uniqueness == 'by_class':
            for p in self._plugins:
                if p.this_class == new_plugin:
                    logger.warning("Plugin '{}' is already loaded . Did not add it.".format(new_plugin.__name__))
                    return

        plugin_instance = new_plugin(self.g_pool, **args)
        if not plugin_instance.alive:
            logger.warning("plugin failed to initialize")
            return

        self._plugins.append(plugin_instance)
        self._plugins.sort(key=lambda p: p.order)

        if self.g_pool.app in ("capture", "player"):
            plugin_instance.init_ui()

    def clean(self):
        '''
        plugins may flag themselves as dead or are flagged as dead. We need to remove them.
        '''
        for p in self._plugins[::-1]:
            if not p.alive:
                if self.g_pool.app in ("capture", "player"):
                    p.deinit_ui()
                p.cleanup()
                logger.debug("Unloaded Plugin: {}".format(p))
                self._plugins.remove(p)

    def get_initializers(self):
        initializers = []
        for p in self._plugins:
            try:
                p_initializer = p.class_name, p.get_init_dict()
                initializers.append(p_initializer)
            except NotImplementedError:
                # not all plugins want to be savable, they will not have the init dict.
                # any object without a get_init_dict method will throw this exception.
                pass
        return initializers


def import_runtime_plugins(plugin_dir):
    """
    Parse all files and folders in 'plugin_dir' and try to import them as modules:
    files are imported if their extension is .py or .so or .dylib
    folder are imported if they contain an __init__.py file

    once a module is sucessfully imported any classes that are subclasses of Plugin
    are added to the runtime plugins list

    any exceptions that are raised during parsing, import filtering and addition are silently ignored.
    """

    runtime_plugins = []
    if os.path.isdir(plugin_dir):
        # we prepend to give the plugin dir content precendece
        # over other modules with identical name.
        sys.path.insert(0, plugin_dir)
        for d in os.listdir(plugin_dir):
            logger.debug('Scanning: {}'.format(d))
            try:
                if os.path.isfile(os.path.join(plugin_dir, d)):
                    d, ext = d.rsplit(".", 1)
                    if ext not in ('py', 'so', 'dylib'):
                        continue
                module = importlib.import_module(d)
                logger.debug('Imported: {}'.format(module))
                for name in dir(module):
                    member = getattr(module, name)
                    if isinstance(member, type) and issubclass(member, Plugin) and member.__name__ != 'Plugin':
                        logger.info('Added: {}'.format(member))
                        runtime_plugins.append(member)
            except Exception as e:
                logger.warning("Failed to load '{}'. Reason: '{}' ".format(d, e))
    return runtime_plugins


# Base plugin definitons
class Visualizer_Plugin_Base(Plugin):
    pass


class Analysis_Plugin_Base(Plugin):
    pass


class Producer_Plugin_Base(Plugin):
    pass


class System_Plugin_Base(Plugin):
    pass


class Experimental_Plugin_Base(Plugin):
    pass
