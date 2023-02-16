"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import importlib
import logging
import os
import sys
import types
from time import time

logger = logging.getLogger(__name__)
"""
A simple example Plugin: 'display_recent_gaze.py'
It is a good starting point to build your own plugin.
"""


class Plugin:
    """docstring for Plugin
    plugin is a base class
    it has all interfaces that will be called
    instances of this class usually get added to a plugins list
    this list will have its members called with all methods invoked.

    """

    # if you have a plugin that can exist multiple times make this false in your derived class
    uniqueness = "by_class"
    # uniqueness = 'not_unique'
    # uniqueness = 'by_base_class'

    # between 0 and 1 this indicated where in the plugin excecution order you plugin lives:
    # <.5  are things that add/mofify information that will be used by other plugins and rely on untouched data.
    # You should not edit frame if you are here!
    # == .5 is the default.
    # >.5 are things that depend on other plugins work like display , saving and streaming
    # you can change this in __init__ for your instance or in the class definition
    order = 0.5
    alive = True

    # menu icon font, possible values `roboto`, `opensans`, `pupil_icons`,
    # or custom loaded font name
    icon_font = "roboto"
    icon_chr = "?"  # character shown in menu icon
    # icon placement and sizing:
    # icon_pos_delta: (x, y) offset relative to default position
    # icon_size_delta: negative values decrease font size, positive values increase it
    # icon_line_height: distance between lines, only relevant for multi-line labels
    icon_pos_delta = (0, 0)
    icon_size_delta = 0
    icon_line_height = 1.0

    def __init__(self, g_pool):
        self.g_pool = g_pool

        if getattr(g_pool, "debug", False):
            self.__monkeypatch_gl_display_error_checking()

    def init_ui(self):
        """
        Called when the context will have a gl window with us. You can do your init for that here.
        """
        pass

    def recent_events(self, events):
        """
        Called in Player and Capture.
        Gets called once every frame.
        If you plan to update data inplace, note that this will affect all plugins executed after you.
        Use self.order to deal with this appropriately
        """
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

        Return True if the event was consumed and should not be propagated
        to any other plugin.
        """
        return False

    def on_pos(self, pos):
        """
        Gets called when the user moves the mouse in the window screen.
        """
        pass

    def on_key(self, key, scancode, action, mods):
        """
        Gets called on key events that were not consumed by the GUI.

        Return True if the event was consumed and should not be propagated
        to any other plugin.

        See http://www.glfw.org/docs/latest/input_guide.html#input_key for
        more information key events.
        """
        return False

    def on_char(self, character):
        """
        Gets called on char events that were not consumed by the GUI.

        Return True if the event was consumed and should not be propagated
        to any other plugin.

        See http://www.glfw.org/docs/latest/input_guide.html#input_char for
        more information char events.
        """
        return False

    def on_drop(self, paths):
        """
        Gets called on dropped paths of files and/or directories on the window.

        Return True if the event was consumed and should not be propagated
        to any other plugin.

        See http://www.glfw.org/docs/latest/input_guide.html#path_drop for
        more information.
        """
        return False

    def on_window_resize(self, window, w, h):
        """
        gets called when user resizes window.
        window is the glfw window handle of the resized window.
        """
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
        """
        Called when the context will have a ui with window. You can do your deinit for that here.
        """
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
        if self.g_pool.app == "exporter":
            if notification.get("delay", 0):
                notification["_notify_time_"] = time() + notification["delay"]
                self.g_pool.delayed_notifications[
                    notification["subject"]
                ] = notification
            else:
                self.g_pool.notifications.append(notification)
        else:
            logger.debug(
                f"'{notification['subject']}' notification sent with keys: "
                f"{tuple(notification.keys())}"
            )
            self.g_pool.ipc_pub.notify(notification)

    @property
    def this_class(self):
        """
        this instance's class
        """
        return self.__class__

    @property
    def class_name(self):
        """
        name of this instance's class
        """
        return self.__class__.__name__

    @classmethod
    def base_class(cls):
        """
        rightmost base class of this class
        this way you can inherit from muliple classes and use the rightmost as your plugin group classifier
        """
        return cls.__bases__[-1]

    @classmethod
    def base_class_name(cls):
        """
        base class name of this class
        """
        return cls.base_class().__name__

    @property
    def pretty_class_name(self):
        return self.__class__.parse_pretty_class_name()

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return cls.__name__.replace("_", " ")

    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        """
        Returns `True` if the plugin class is available within the `g_pool` context.
        """
        return True

    def add_menu(self):
        """
        This fn is called when the plugin ui is initialized. Do not change!
        """
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
        self.menu = ui.Growing_Menu("Unnamed Menu", header_pos="headline")
        if self.uniqueness == "not_unique":
            self.menu.append(ui.Button("Close", close))
        self.menu_icon = ui.Icon(
            "collapsed",
            self.menu,
            label=self.icon_chr,
            label_font=self.icon_font,
            label_offset_size=self.icon_size_delta,
            label_offset_x=self.icon_pos_delta[0],
            label_offset_y=self.icon_pos_delta[1],
            label_line_height=self.icon_line_height,
            on_val=False,
            off_val=True,
            setter=toggle_menu,
        )
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

    @property
    def ui_available(self):
        try:
            return self.menu is not None
        except AttributeError:
            return False

    def __monkeypatch_gl_display_error_checking(self):
        # Monkeypatch gl_display functions to include error checking. This is because we
        # often receive OpenGL errors as results of buggy pyglui code that gets called
        # in gl_display. Since pyglui does not check glGetError(), we run into these
        # errors at other places at the code when e.g. using pyopengl. By checking after
        # every plugin we can at least partially localize the error!

        # Take gl_display function prototype from class, i.e. not bound to an
        # instance. This will return potentially overwritten implementations
        # from child classes.
        unpatched_gl_display = self.__class__.gl_display

        from OpenGL.GL import glGetError
        from OpenGL.GLU import gluErrorString

        # Create wrapper method including a glGetError check
        def wrapper(_self):
            unpatched_gl_display(_self)
            err = glGetError()
            if err != 0:
                logger.error(
                    f"Encountered OpenGL Error in Plugin '{_self.class_name}'!"
                    f" Error code: {err}, msg: {gluErrorString(err)}"
                )

        # Bind wrapper to current instance
        self.gl_display = types.MethodType(wrapper, self)


# Plugin manager classes and fns
class Plugin_List:
    """This is the Plugin Manager
    It is a self sorting list with a few functions to manage adding and
    removing Plugins and lacking most other list methods.
    """

    def __init__(self, g_pool, plugin_initializers):
        self._plugins = []
        self.g_pool = g_pool
        plugin_by_name = g_pool.plugin_by_name

        # add self as g_pool.plguins object to allow plugins to call the plugins list
        # during init. this will be done again when the init returns but is kept there
        # for readablitly.
        self.g_pool.plugins = self

        # NOTE: we should not .add() plugins if they get removed immediately again
        # because of uniqueness constraints. Here we are filtering the passed list first
        # before calling .add(). This is important so we e.g. don't initialize the UVC
        # source in player, which will try installing drivers and crash in the bundle.

        # expand first for later filtering
        expanded_initializers = []
        for name, args in plugin_initializers:
            try:
                expanded_initializers.append((plugin_by_name[name], name, args))
            except KeyError:
                logger.debug(f"Plugin {name} failed to load, not available for import.")

        expanded_initializers.sort(key=lambda data: data[0].order)

        # skip plugins that are not available within g_pool context
        # not removing them here will break the uniqueness logic bellow
        expanded_initializers = [
            (plugin, name, args)
            for (plugin, name, args) in expanded_initializers
            if plugin.is_available_within_context(self.g_pool)
        ]

        # only add plugins that won't be replaced by newer plugins
        for i, (plugin, name, args) in enumerate(expanded_initializers):
            for new_plugin, new_name, _ in expanded_initializers[i + 1 :]:
                if (
                    new_plugin.uniqueness == "by_base_class"
                    and plugin.base_class() == new_plugin.base_class()
                ) or (new_plugin.uniqueness == "by_class" and plugin == new_plugin):
                    logger.debug(
                        f"Skipping initialization of plugin {name} because it will be"
                        f" replaced by newer plugin {new_name} with uniqueness"
                        f" `{new_plugin.uniqueness}`."
                    )
                    break
            else:
                # no new_plugin found which would replace old_plugin
                logger.debug(f"Loading plugin: {name} with settings {args}")
                self.add(plugin, args)

    def __iter__(self):
        yield from self._plugins

    def __str__(self):
        return f"Plugin List: {self._plugins}"

    def add(self, new_plugin_cls, args={}):
        """
        add a plugin instance to the list.
        """

        # Check if the plugin class is supported within the current g_pool context
        if not new_plugin_cls.is_available_within_context(self.g_pool):
            logger.debug(
                f"Plugin {new_plugin_cls.__name__} not available; skip adding it to plugin list."
            )
            return

        self._find_and_remove_duplicates(new_plugin_cls)

        plugin_instance = new_plugin_cls(self.g_pool, **args)
        if not plugin_instance.alive:
            logger.warning(f"Plugin {new_plugin_cls.__name__} failed to initialize")
            return

        self._plugins.append(plugin_instance)
        self._plugins.sort(key=lambda p: p.order)

        if self.g_pool.app in ("capture", "player") or "eye" in self.g_pool.process:
            plugin_instance.init_ui()

    def _find_and_remove_duplicates(self, new_plugin_cls):
        for duplicate in self._duplicates(new_plugin_cls):
            self._remove_duplicated_instance(duplicate)

    def _duplicates(self, new_plugin_cls):
        if new_plugin_cls.uniqueness == "by_base_class":
            yield from self._duplicates_by_rule(
                self._has_same_base_class, new_plugin_cls
            )
        elif new_plugin_cls.uniqueness == "by_class":
            yield from self._duplicates_by_rule(self._is_same_class, new_plugin_cls)

    def _duplicates_by_rule(self, is_duplicate_rule, new_plugin_cls):
        duplicates = (
            old_plugin_inst
            for old_plugin_inst in self._plugins
            if is_duplicate_rule(old_plugin_inst, new_plugin_cls)
        )
        yield from duplicates

    def _remove_duplicated_instance(self, duplicated_plugin_inst):
        name = duplicated_plugin_inst.pretty_class_name
        uniq = duplicated_plugin_inst.uniqueness
        message = f"Replacing {name} due to '{uniq}' uniqueness"
        logger.debug(message)
        duplicated_plugin_inst.alive = False
        self.clean()

    @staticmethod
    def _has_same_base_class(old_plugin_inst, new_plugin_cls):
        return old_plugin_inst.base_class() == new_plugin_cls.base_class()

    @staticmethod
    def _is_same_class(old_plugin_inst, new_plugin_cls):
        return old_plugin_inst.this_class == new_plugin_cls

    def clean(self):
        """
        plugins may flag themselves as dead or are flagged as dead. We need to remove them.
        """
        for p in self._plugins[::-1]:
            if not p.alive:
                if self.g_pool.app in ("capture", "player") or self.g_pool.process in (
                    "eye0",
                    "eye1",
                ):
                    p.deinit_ui()
                p.cleanup()
                logger.debug(f"Unloaded Plugin: {p}")
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
    logger.debug(f"Searching {plugin_dir}...")
    if os.path.isdir(plugin_dir):
        # we prepend to give the plugin dir content precendece
        # over other modules with identical name.
        sys.path.insert(0, plugin_dir)
        for d in os.listdir(plugin_dir):
            logger.debug(f"Scanning: {d}")
            try:
                if os.path.isfile(os.path.join(plugin_dir, d)):
                    d, ext = d.rsplit(".", 1)
                    if ext not in ("py", "so", "dylib"):
                        continue
                module = importlib.import_module(d)
                logger.debug(f"Imported: {module}")
                for name in dir(module):
                    member = getattr(module, name)
                    if (
                        isinstance(member, type)
                        and issubclass(member, Plugin)
                        and member.__name__ != "Plugin"
                    ):
                        logger.debug(f"Added: {member}")
                        runtime_plugins.append(member)
            except Exception as e:
                logger.warning(f"Failed to load '{d}'. Reason: '{e}' ")
                import traceback

                logger.debug(traceback.format_exc())
    else:
        logger.debug(f"{plugin_dir} is not a directory. Skipping imports!")
    return runtime_plugins


class System_Plugin_Base(Plugin):
    pass
