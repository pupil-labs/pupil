#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
'''
The  atb  module  provides  bindings  for  AntTweakBar which  is  a  small  and
easy-to-use C/C++  library that allows programmers  to quickly add  a light and
intuitive graphical  user interface into graphic applications  based on OpenGL,
DirectX 9 or DirectX 10 to interactively tweak their parameters on-screen.
'''
import ctypes
from raw import *

def check_error(status, error=0):
    if status == error:
        raise Exception(TwGetLastError())
    else:
        return status

def enum(name, values):
    E = (TwEnumVal*len(values))()
    for i,(label,value) in enumerate(values.iteritems()):
        E[i].Value, E[i].Label = value, label
    return check_error(TwDefineEnum(name, E, len(values)))

def init():
    check_error(TwInit(TW_OPENGL, 0))

# handets in __del__
# def shutdown():
#     check_error( TwTerminate() )

def _dict_to_defs(args):
    '''
    Converts a dictionary like {a:'b', 1:2} to the string "a=b 1=2" suitable
    for passing to define method. Automatic type conversion is done as follows:
    - if the value is bool result is simply the name of the string eg
        {'closed':True} -> "closed"
    - if the value is a tuple the items are converted to strings and joined
        by spaces, eg {'size':(10, 20)} -> "size='10 20'"
    '''
    r = []
    for k, v in args.iteritems():
        if type(v) is bool:    v = ""
        elif type(v) is tuple: v = "='%s'" % " ".join((str(i) for i in v))
        else:                  v = "='%s'" % str(v)
        r.append(k+v)
    return " ".join(r)


class Bar(object):
    '''
    Bar is an internal structure used to store tweak bar attributes and
    states.
    '''
    def __init__(self, name=None, **defs):
        '''
        Create a new bar.

        Arguments:
        ----------
        name : str
            Name of the new bar.

        Keyword arguments:
        ------------------
        label : str
            Changes the label of a bar, that is the title displayed on top of a
            bar. By default, the label is the name used when the bar was
            created.

        help : str
            Defines the help message associated to a bar. This message will be
            displayed inside the Help bar automatically created to help the
            user.

            You can also define a global help message. It will be displayed at
            the beginning of the Help bar. To define it, use the GLOBAL keyword
            instead of the bar name.

        color : (int,int,int)
            Changes the color of a bar to (red,green,blue).

            red, green and blue are integer values between 0 and 255 that
            define the red, green and blue color channels. See also the alpha
            and text parameters to change bar visual aspect.

        alpha : int
            Changes the bar opacity.

            Bar opacity can vary from 0 for fully transparent to 255 for fully
            opaque. See also the color and text parameters to change bar visual
            aspect.

        text : str
            Changes text color to 'dark' or 'light'.

            Depending on your application background color and on bar color and
            alpha, bar text might be more readable if it is dark or light. This
            parameter allows to switch between the two modes. See also the
            color and alpha parameters to change bar visual aspect.

        position : (int,int)
            Move a bar to a new position (x,y).

            x and y are positive integer values that represent the new position
            of the bar in pixels. (x=0, y=0) is upper-left corner of the
            application window.

        size : (int,int)
            Change the bar size to (sx,sy).

            sx and sy are positive integer values that represent the new size
            of the bar in pixels.

        valueswidth : int
            Change the width 'w' of the bar right column used to display numerical
            values.

            w is a positive integer that represents width in pixels.

        refresh : float
            Change the refresh rate 'r' of the bar.

            Values displayed by a bar are automatically updated to reflect
            changes of their associated variables. r is a real value
            corresponding to the number of seconds between two updates.

        fontsize : int
            Change the size 's' of the font used by the bars.

            s is 1 for small font, 2 for medium font, or 3 for large font. Note
            that all bars share the same font, so this change is applied to all
            bars.

        visible : bool
            Show or hide a tweak bar.

        iconified : bool
            Iconify or deiconify a tweak bar.

        iconpos : str
            Changes the place where icons of iconified bars are displayed.

            p is one of the following values:
            - 'bottomleft' or 'bl' for bottom-left corner of the window (default).
            - 'bottomright' or 'br' for bottom-right corner of the window.
            - 'topleft' or 'tl' for top-left corner of the window.
            - 'topright' or 'tr' for top-right corner of the window.

            Note that this parameter is applied to all bar icons.

        iconalign : str
            Changes the alignment of icons of iconified bars. It can be
            'vertical' (the default), or 'horizontal'.

            Note that this parameter is applied to all bar icons.

        iconmargin : (int,int)
            Add a margin (x,y) between borders of the window and icons of
            iconified bars. x and y are the number of pixels between window
            borders and icons in the x and y directions respectively.

            Note that this parameter is applied to all bar icons.

        iconifiable : bool
            Allow a bar to be iconified or not by the user.

        movable : bool
            Allow a bar to be moved or not by the user.

        resizable : bool
            Allow a bar to be resized or not by the user.

        fontresizable : bool
            Allow bar fonts to be resized or not by the user.

            Note that this parameter is applied to all bars.

        alwaystop : bool
            Set a bar to be always on top of the others.

        alwaysbottom : bool
            Set a bar to be always behind the others.
        '''

        if not name:
            name = "Unnamed"
        self._name = name
        self._bar = TwNewBar(name)
        if defs:
            self.define(_dict_to_defs(defs))
        self._c_callbacks = []


    def _get_name(self):
        return self._name
    name = property(_get_name,
                    doc='''Name of the bar''')


    def _set_label(self, label):
        c = ctypes.c_char_p(label)
        TwSetParam(self._bar, "", "label", PARAM_CSTRING, 1, c)
    def _get_label(self):
        c = ctypes.create_string_buffer(4096)
        TwGetParam(self._bar, "", "label", PARAM_CSTRING, 4095, c)
        return c.value
    label = property(_get_label, _set_label,
                     doc='''Bar label.

    Changes the label of a bar, that is the title displayed on top of a bar.
    By default, the label is the name used when the bar was created.

    :type: str
    ''')



    def _set_alpha(self, alpha):
        c = ctypes.c_int(alpha)
        TwSetParam(self._bar, "", "alpha", PARAM_INT32, 1, ctypes.byref(c))
    def _get_alpha(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "alpha", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    alpha = property(_get_alpha, _set_alpha,
                     doc='''Bar opacity.

    Bar opacity can vary from 0 for fully transparent to 255 for fully opaque.
    See also the color and text parameters to change bar visual aspect.

    :type: int
    ''')



    def _set_color(self, color):
        c = (ctypes.c_int*3)(color[0],color[1],color[2])
        TwSetParam(self._bar, "", "color", PARAM_INT32, 3, ctypes.byref(c))
    def _get_color(self):
        c = (ctypes.c_int*3)(0,0,0)
        TwGetParam(self._bar, "", "color", PARAM_INT32, 3, ctypes.byref(c))
        return c[0], c[1], c[2]
    color = property(_get_color, _set_color,
                     doc='''Bar color.

    Red, green and blue are integer values between 0 and 255 that define the
    red, green and blue color channels. See also the alpha and text parameters
    to change bar visual aspect.

    :type: (int,int,int)
    ''')



    def _set_help(self, help):
        c = ctypes.c_char_p(help)
        TwSetParam(self._bar, "", "help", PARAM_CSTRING, 1, c)
    def _get_help(self):
        c = ctypes.create_string_buffer(4096)
        TwGetParam(self._bar, "", "help", PARAM_CSTRING, 4095, c)
        return c.value
    help = property(_get_help, _set_help,
                     doc='''Help message.

    Defines the help message associated to a bar. This message will be
    displayed inside the Help bar automatically created to help the user.

    :type: str
    ''')



    def _set_text(self, text):
        c = ctypes.c_char_p(text)
        TwSetParam(self._bar, "", "text", PARAM_CSTRING, 1, c)
    def _get_text(self):
        c = ctypes.create_string_buffer(16)
        TwGetParam(self._bar, "", "text", PARAM_CSTRING, 15, c)
        return c.value
    text = property(_get_text, _set_text,
                     doc='''Text color.

    Depending on your application background color and on bar color and alpha,
    bar text might be more readable if it is dark or light. This parameter
    allows to switch between the two modes. See also the color and alpha
    parameters to change bar visual aspect.

    :type: str
    ''')



    def _set_position(self, position):
        c = (ctypes.c_int*2)(position[0],position[1])
        TwSetParam(self._bar, "", "position", PARAM_INT32, 2, ctypes.byref(c))
    def _get_position(self):
        c = (ctypes.c_int*2)(0,0)
        TwGetParam(self._bar, "", "position", PARAM_INT32, 2, ctypes.byref(c))
        return c[0], c[1]
    position = property(_get_position, _set_position,
                     doc='''Bar position (x,y).

    x and y are positive integer values that represent the new position of the
    bar in pixels. (x=0, y=0) is upper-left corner of the application window.

    :type: (int,int)
    ''')



    def _set_size(self, size):
        c = (ctypes.c_int*2)(size[0],size[1])
        TwSetParam(self._bar, "", "size", PARAM_INT32, 2, ctypes.byref(c))
    def _get_size(self):
        c = (ctypes.c_int*2)(0,0)
        TwGetParam(self._bar, "", "size", PARAM_INT32, 2, ctypes.byref(c))
        return c[0], c[1]
    size = property(_get_size, _set_size,
                     doc='''Bar size (sx,sy).

    sx and sy are positive integer values that represent the new size of the bar
    in pixels.

    :type: (int,int)
    ''')



    def _set_valuewidth(self, valuewidth):
        c = ctypes.c_int(valuewidth)
        TwSetParam(self._bar, "", "valuewidth", PARAM_INT32, 1, ctypes.byref(c))
    def _get_valuewidth(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "valuewidth", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    valuewidth = property(_get_valuewidth, _set_valuewidth,
                     doc='''Value width.

    Width of the bar right column used to display numerical values.

    :type: int
    ''')



    def _set_fontsize(self, fontsize):
        c = ctypes.c_int(fontsize)
        TwSetParam(self._bar, "", "fontsize", PARAM_INT32, 1, ctypes.byref(c))
    def _get_fontsize(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "fontsize", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    fontsize = property(_get_fontsize, _set_fontsize,
                     doc='''Font size s.

    s is 1 for small font, 2 for medium font, or 3 for large font. Note that
    all bars share the same font, so this change is applied to all bars.

    fontsize is a global parameter.

    :type: int
    ''')



    def _set_refresh(self, refresh):
        c = ctypes.c_float(refresh)
        TwSetParam(self._bar, "", "refresh", PARAM_FLOAT, 1, ctypes.byref(c))
    def _get_refresh(self):
        c = ctypes.c_float(0)
        TwGetParam(self._bar, "", "refresh", PARAM_FLOAT, 1, ctypes.byref(c))
        return c.value
    refresh = property(_get_refresh, _set_refresh,
                     doc='''Refresh rate.

    Values displayed by a bar are automatically updated to reflect changes of
    their associated variables. r is a real value corresponding to the number
    of seconds between two updates.

    :type: float
    ''')



    def _set_visible(self, visible):
        c = ctypes.c_int(visible)
        TwSetParam(self._bar, "", "visible", PARAM_INT32, 1, ctypes.byref(c))
    def _get_visible(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "visible", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    visible = property(_get_visible, _set_visible,
                     doc='''Bar visibility.

    See also show and hide method.

    :type: int
    ''')



    def _set_iconified(self, iconified):
        c = ctypes.c_int(iconified)
        TwSetParam(self._bar, "", "iconified", PARAM_INT32, 1, ctypes.byref(c))
    def _get_iconified(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "iconified", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    iconified = property(_get_iconified, _set_iconified,
                     doc='''Bar iconification.

    Iconify or deiconify the bar.

    :type: int
    ''')



    def _set_iconpos(self, iconpos):
        c = ctypes.c_char_p(iconpos)
        TwSetParam(self._bar, "", "iconpos", PARAM_CSTRING, 1, c)
    def _get_iconpos(self):
        c = ctypes.create_string_buffer(32)
        TwGetParam(self._bar, "", "iconpos", PARAM_CSTRING, 31, c)
        return c.value
    iconpos = property(_get_iconpos, _set_iconpos,
                     doc='''Bar icon position p.

    p is one of the following values:

    - 'bottomleft' or 'bl' for bottom-left corner of the window (default).
    - 'bottomright' or 'br' for bottom-right corner of the window.
    - 'topleft' or 'tl' for top-left corner of the window.
    - 'topright' or 'tr' for top-right corner of the window.

    Note that this parameter is applied to all bar icons.

    :type: str
    ''')



    def _set_iconalign(self, iconalign):
        c = ctypes.c_char_p(iconalign)
        TwSetParam(self._bar, "", "iconalign", PARAM_CSTRING, 1, c)
    def _get_iconalign(self):
        c = ctypes.create_string_buffer(32)
        TwGetParam(self._bar, "", "iconalign", PARAM_CSTRING, 31, c)
        return c.value
    iconalign = property(_get_iconalign, _set_iconalign,
                     doc='''Bar icon alignment p.

    Changes the alignment of icons of iconified bars. It can be 'vertical' (the
    default), or 'horizontal'.

    Note that this parameter is applied to all bar icons.

    :type: str
    ''')



    def _set_iconmargin(self, iconmargin):
        c = (ctypes.c_int*2)(iconmargin[0],iconmargin[1])
        TwSetParam(self._bar, "", "iconmargin", PARAM_INT32, 2, ctypes.byref(c))
    def _get_iconmargin(self):
        c = (ctypes.c_int*2)(0,0)
        TwGetParam(self._bar, "", "iconmargin", PARAM_INT32, 2, ctypes.byref(c))
        return c[0], c[1]
    iconmargin = property(_get_iconmargin, _set_iconmargin,
                     doc='''Bar icon margin (x,y).

     Add a margin between borders of the window and icons of iconified bars. x
     and y are the number of pixels between window borders and icons in the x
     and y directions respectively.

     Note that this parameter is applied to all bar icons.

    :type: (int,int)
    ''')



    def _set_iconifiable(self, iconifiable):
        c = ctypes.c_int(iconifiable)
        TwSetParam(self._bar, "", "iconifiable", PARAM_INT32, 1, ctypes.byref(c))
    def _get_iconifiable(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "iconifiable", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    iconifiable = property(_get_iconifiable, _set_iconifiable,
                     doc='''Allow a bar to be iconified or not by the user.

    :type: int
    ''')



    def _set_movable(self, movable):
        c = ctypes.c_int(movable)
        TwSetParam(self._bar, "", "movable", PARAM_INT32, 1, ctypes.byref(c))
    def _get_movable(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "movable", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    movable = property(_get_movable, _set_movable,
                     doc='''Allow a bar to be moved or not by the user.

    :type: int
    ''')



    def _set_resizable(self, resizable):
        c = ctypes.c_int(resizable)
        TwSetParam(self._bar, "", "resizable", PARAM_INT32, 1, ctypes.byref(c))
    def _get_resizable(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "resizable", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    resizable = property(_get_resizable, _set_resizable,
                     doc='''Allow a bar to be resized or not by the user.

    :type: int
    ''')



    def _set_fontresizable(self, fontresizable):
        c = ctypes.c_int(fontresizable)
        TwSetParam(self._bar, "", "fontresizable", PARAM_INT32, 1, ctypes.byref(c))
    def _get_fontresizable(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "fontresizable", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    fontresizable = property(_get_fontresizable, _set_fontresizable,
                     doc='''Allow a bar to have font resized or not by the user.

    :type: int
    ''')



    def _set_alwaystop(self, alwaystop):
        c = ctypes.c_int(alwaystop)
        TwSetParam(self._bar, "", "alwaystop", PARAM_INT32, 1, ctypes.byref(c))
    def _get_alwaystop(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "alwaystop", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    alwaystop = property(_get_alwaystop, _set_alwaystop,
                     doc='''Set a bar to be always on top of the others.

    :type: int
    ''')



    def _set_alwaybottom(self, alwaybottom):
        c = ctypes.c_int(alwaybottom)
        TwSetParam(self._bar, "", "alwaybottom", PARAM_INT32, 1, ctypes.byref(c))
    def _get_alwaybottom(self):
        c = ctypes.c_int(0)
        TwGetParam(self._bar, "", "alwaybottom", PARAM_INT32, 1, ctypes.byref(c))
        return c.value
    alwaybottom = property(_get_alwaybottom, _set_alwaybottom,
                     doc='''Set a bar to be always behind the others.

    :type: int
    ''')


    def __del__(self):
        check_error(TwDeleteAllBars(self._bar))
        check_error(TwTerminate())

    def draw(self):
        check_error(TwDraw())

    def clear(self):
        check_error(TwRemoveAllVars(self._bar))

    def remove(self, name):
        check_error(TwRemoveVar(self._bar, name))

    def update(self):
        check_error(TwRefreshBar(self._bar))

    def bring_to_front(self):
        check_error(TwSetTopBar(self._bar))

    def add_var(self, name, value=None, vtype=None, readonly=False,
                getter=None, setter=None, data=None, **defs):
        '''
        Add a new variable to the tweak bar.

        Arguments:
        ----------
        name : str
            The name of the variable. It will be displayed in the tweak bar if
            no label is specified for this variable. It will also be used to
            refer to this variable in other functions, so choose a unique,
            simple and short name and avoid special characters like spaces or
            punctuation marks.
        value : ctypes
           Value of the variable
        vtype : TYPE_xxx
           Type of the variable. It must be one of the TYPE_xxx constants or an enum type.
        readonly: bool
            Makes a variable read-only or read-write. The user would be able to
            modify it or not.
        getter : func(data) or func()
            The callback function that will be called by AntTweakBar to get the
            variable's value.
        setter : func(value, data)
           The callback function that will be called to change the variable's
           value.
        data : object
           Data to be send to getter/setter functions

        Keyword arguments:
        ------------------
        label : str
            Changes the label of a variable, that is the name displayed before
            its value. By default, the label is the name used when the variable
            was added to a bar.
        help : str
            Defines the help message associated to a variable. This message will
            be displayed inside the Help bar automatically created to help the
            user.
        group : str
            Move a variable into a group. This allows you to regroup
            variables. If groupname does not exist, it is created and added to
            the bar. You can also put groups into groups, and so obtain a
            hierarchical organization.
        visible: bool
            Show or hide a variable.
        min / max: scalar
            Set maximum and minimum value of a variable. Thus, user cannot
            exceed these bounding values when (s)he edit the variable.
        step: scalar
            Set a step value for a variable. When user interactively edit the
            variable, it is incremented or decremented by this value.
        precision : scalar
            Defines the number of significant digits printed after the period
            for floating point variables. This number must be between 0 and 12,
            or -1 to disable precision and use the default formating.
            If precision is not defined and step is defined, the step number of
            significant digits is used for defining the precision.
        hexa : bool
            For integer variables only.
            Print an integer variable as hexadecimal or decimal number.
        True / False : str
            For boolean variables only.
            By default, if a boolean variable is true, it is displayed as 'ON',
            and if it is false, as 'OFF'. You can change this message with the
            true and false parameters, the new string will replace the previous
            message.
        opened : bool
            For groups only.
            Fold or unfold a group displayed in a tweak bar (as when the +/-
            button displayed in front of the group is clicked).

        '''

        groups = name.split('/')
        name = groups[-1]
        _typemap = {ctypes.c_bool:      TW_TYPE_BOOL8,
                    ctypes.c_int:       TW_TYPE_INT16,
                    ctypes.c_long:      TW_TYPE_INT32,
                    ctypes.c_float:     TW_TYPE_FLOAT,
                    ctypes.c_float * 3: TW_TYPE_COLOR3F,
                    ctypes.c_float * 4: TW_TYPE_COLOR4F}
        _typemap_inv = dict([(v, k) for k, v in _typemap.iteritems()])

        if vtype is None and value is not None:
            vtype = _typemap.get(type(value))
        elif vtype:
            vtype = _typemap.get(vtype, vtype)
        elif vtype is None and getter is not None:
            t = type(getter())
            if t  == bool:
                vtype = TW_TYPE_BOOL8
            elif t == int:
                vtype = TW_TYPE_INT16
            elif t == long:
                vtype = TW_TYPE_INT32
            elif t == float:
                vtype = TW_TYPE_FLOAT
        else:
            raise ValueError("Cannot determin value type")
        ctype = _typemap_inv.get(vtype,c_int)
        def_str = _dict_to_defs(defs)

        if getter:
            def wrapped_getter(p, user_data):
                v = ctypes.cast(p, ctypes.POINTER(ctype))
                d = ctypes.cast(user_data, ctypes.py_object)
                if d.value is not None:
                    v[0] = getter(d.value)
                else:
                    v[0] = getter()

        if setter:
            def wrapped_setter(p, user_data):
                v = ctypes.cast(p, ctypes.POINTER(ctype))
                d = ctypes.cast(user_data, ctypes.py_object)
                if d.value is not None:
                    setter(v[0], d.value)
                else:
                    setter(v[0])

        if (getter and readonly) or (getter and not setter):
            c_callback = GET_FUNC(wrapped_getter)
            self._c_callbacks.append(c_callback)
            r = TwAddVarCB(self._bar, name, vtype, None, c_callback,
                           ctypes.py_object(data), def_str)
        elif (getter and setter):
            c_setter = SET_FUNC(wrapped_setter)
            c_getter = GET_FUNC(wrapped_getter)
            self._c_callbacks.extend((c_setter, c_getter))
            r = TwAddVarCB(self._bar, name, vtype, c_setter, c_getter,
                           ctypes.py_object(data), def_str)
        elif readonly:
            r = TwAddVarRO(self._bar, name, vtype, ctypes.byref(value), def_str)
        else:
            r = TwAddVarRW(self._bar, name, vtype, ctypes.byref(value), def_str)

        check_error(r)
        if len(groups) > 1:
            name = self.name
            for i in range(len(groups)-1,0,-1):
                self.define("group=%s" % groups[i-1], groups[i])


    def add_button(self, name, callback, data=None, **defs):
        '''
        '''
        def wrapped_callback(userdata):
            d = ctypes.cast(userdata, ctypes.py_object)
            if d.value is not None:
                callback(d.value)
            else:
                callback()
        c_callback = BUTTON_FUNC(wrapped_callback)
        self._c_callbacks.append(c_callback)
        def_str = _dict_to_defs(defs)
        data_p = ctypes.py_object(data)
        check_error( TwAddButton(self._bar, name, c_callback, data_p, def_str) )


    def add_separator(self, name, **defs):
        ''' '''
        def_str = _dict_to_defs(defs)
        check_error( TwAddSeparator(self._bar, name, def_str ) )


    def define(self, definition='', varname=None):
        '''
        This function defines optional parameters for tweak bars and
        variables. For instance, it allows you to change the color of a tweak
        bar, to set a min and a max value for a variable, to add an help
        message that inform users of the meaning of a variable, and so on...

        If no varname is given, definition is applied to bar, else, it is
        applied to the given variable.
        '''
        if varname:
            arg = '%s/%s %s' % (self.name, varname,definition)
        else:
            arg = '%s %s' % (self.name, definition)
        check_error(TwDefine(arg))
