from glfw import *


if __name__ == '__main__':
    import sys

    keyrepeat  = False
    systemkeys = True
    counter    = 0
    keynames   = {
        GLFW_KEY_UNKNOWN:      "unknown",
        GLFW_KEY_SPACE:        "space",
        GLFW_KEY_ESC:          "escape",
        GLFW_KEY_F1:           "F1",
        GLFW_KEY_F2:           "F2",
        GLFW_KEY_F3:           "F3",
        GLFW_KEY_F4:           "F4",
        GLFW_KEY_F5:           "F5",
        GLFW_KEY_F6:           "F6",
        GLFW_KEY_F7:           "F7",
        GLFW_KEY_F8:           "F8",
        GLFW_KEY_F9:           "F9",
        GLFW_KEY_F10:          "F10",
        GLFW_KEY_F11:          "F11",
        GLFW_KEY_F12:          "F12",
        GLFW_KEY_F13:          "F13",
        GLFW_KEY_F14:          "F14",
        GLFW_KEY_F15:          "F15",
        GLFW_KEY_F16:          "F16",
        GLFW_KEY_F17:          "F17",
        GLFW_KEY_F18:          "F18",
        GLFW_KEY_F19:          "F19",
        GLFW_KEY_F20:          "F20",
        GLFW_KEY_F21:          "F21",
        GLFW_KEY_F22:          "F22",
        GLFW_KEY_F23:          "F23",
        GLFW_KEY_F24:          "F24",
        GLFW_KEY_F25:          "F25",
        GLFW_KEY_UP:           "up",
        GLFW_KEY_DOWN:         "down",
        GLFW_KEY_LEFT:         "left",
        GLFW_KEY_RIGHT:        "right",
        GLFW_KEY_LSHIFT:       "left shift",
        GLFW_KEY_RSHIFT:       "right shift",
        GLFW_KEY_LCTRL:        "left control",
        GLFW_KEY_RCTRL:        "right control",
        GLFW_KEY_LALT:         "left alt",
        GLFW_KEY_RALT:         "right alt",
        GLFW_KEY_TAB:          "tab",
        GLFW_KEY_ENTER:        "enter",
        GLFW_KEY_BACKSPACE:    "backspace",
        GLFW_KEY_INSERT:       "insert",
        GLFW_KEY_DEL:          "delete",
        GLFW_KEY_PAGEUP:       "page up",
        GLFW_KEY_PAGEDOWN:     "page down",
        GLFW_KEY_HOME:         "home",
        GLFW_KEY_END:          "end",
        GLFW_KEY_KP_0:         "keypad 0",
        GLFW_KEY_KP_1:         "keypad 1",
        GLFW_KEY_KP_2:         "keypad 2",
        GLFW_KEY_KP_3:         "keypad 3",
        GLFW_KEY_KP_4:         "keypad 4",
        GLFW_KEY_KP_5:         "keypad 5",
        GLFW_KEY_KP_6:         "keypad 6",
        GLFW_KEY_KP_7:         "keypad 7",
        GLFW_KEY_KP_8:         "keypad 8",
        GLFW_KEY_KP_9:         "keypad 9",
        GLFW_KEY_KP_DIVIDE:    "keypad divide",
        GLFW_KEY_KP_MULTIPLY:  "keypad multiply",
        GLFW_KEY_KP_SUBTRACT:  "keypad subtract",
        GLFW_KEY_KP_ADD:       "keypad add",
        GLFW_KEY_KP_DECIMAL:   "keypad decimal",
        GLFW_KEY_KP_EQUAL:     "keypad equal",
        GLFW_KEY_KP_ENTER:     "keypad enter",
        GLFW_KEY_KP_NUM_LOCK:  "keypad num lock",
        GLFW_KEY_CAPS_LOCK:    "caps lock",
        GLFW_KEY_SCROLL_LOCK:  "scroll lock",
        GLFW_KEY_PAUSE:        "pause",
        GLFW_KEY_LSUPER:       "left super",
        GLFW_KEY_RSUPER:       "right super",
        GLFW_KEY_MENU:         "menu"
    }

    def get_key_name( key ):
        global keynames

        if key in keynames.keys():
            return keynames[key]
        return ""
    
    def get_action_name( action ):
        if action == GLFW_PRESS:
            return "was pressed"
        elif action == GLFW_RELEASE:
            return "was released"
        return "caused unknown action"

    def get_button_name( button ):
        if button == GLFW_MOUSE_BUTTON_LEFT:
            return "left"
        elif button == GLFW_MOUSE_BUTTON_RIGHT:
            return "right"
        elif button == GLFW_MOUSE_BUTTON_MIDDLE:
            return "middle";
        return ""

    def get_character_string( character ):
        result = c_char_p("       ")
        length = wctomb(result,character)
        if length == -1:
            return ""
        return str(result.value)[:length]

    def window_size_callback( width, height ):
        global counter

        print "%08x at %0.3f: Window size: %i %i" % \
            (counter, glfwGetTime(), width, height);
        counter += 1
        glViewport(0, 0, width, height)

    def window_close_callback():
        global counter

        print "%08x at %0.3f: Window close" % (counter, glfwGetTime())
        counter += 1
        return 1

    def window_refresh_callback():
        global counter

        print "%08x at %0.3f: Window refresh" % (counter, glfwGetTime())
        counter += 1

    def mouse_button_callback( button, action ):
        global counter

        name = get_button_name( button )
        print "%08x at %0.3f: Mouse button %i" % (counter, glfwGetTime(), button),
        if ( name ):
            print "(%s) was %s" % (name, get_action_name( action ))
        else:
            print "was %s\n" % get_action_name( action )
        counter += 1

    def mouse_position_callback( x, y ):
        global counter

        print "%08x at %0.3f: Mouse position: %i %i" % (counter, glfwGetTime(), x, y)
        counter += 1

    def mouse_wheel_callback( position ):
        global counter

        print "%08x at %0.3f: Mouse wheel: %i" % (counter, glfwGetTime(), position)
        counter += 1

    def key_callback( key, action ):
        global counter, keyrepeat, systemkeys

        name = get_key_name(key)
        print "%08x at %0.3f: Key 0x%04x" % (counter, glfwGetTime(), key),
        counter +=1
        if name:
            print "(%s) was %s" % (name, get_action_name(action))
        elif isgraph( key ):
            print "(%c) was %s" % (key, get_action_name(action))
        else:
            print "was %s" % get_action_name(action)
        if action != GLFW_PRESS:
            return
        if key == 'R':
            keyrepeat = not keyrepeat;
            if keyrepeat:
                glfwEnable( GLFW_KEY_REPEAT )
            else:
                glfwDisable( GLFW_KEY_REPEAT )
            print "Key repeat",
            if keyrepeat: print "enabled"
            else:         print "disabled"
        elif key == 'S':
            systemkeys =  not systemkeys;
            if systemkeys:
                glfwEnable( GLFW_SYSTEM_KEYS )
            else:
                glfwDisable( GLFW_SYSTEM_KEYS )
            print "System keys",
            if systemkeys: print "enabled"
            else:          print "disabled"


    def char_callback( character, action ):
        global counter

        print "%08x at %0.3f: Character 0x%04x" % (counter, glfwGetTime(), character),
        print " (%s) %s" %(get_character_string(character), get_action_name(action))
        counter += 1

    if not glfwInit():
        sys.exit()

    if not glfwOpenWindow( 0,0,0,0,0,0,0,0, GLFW_WINDOW ):
        glfwTerminate()
        print 'Failed to open GLFW default window'
        sys.exit()

    glfwSetWindowTitle("Event Linter")
    glfwSwapInterval(1)
    glfwSetWindowSizeCallback(window_size_callback)
    glfwSetWindowCloseCallback(window_close_callback)
    glfwSetWindowRefreshCallback(window_refresh_callback)
    glfwSetMouseButtonCallback(mouse_button_callback)
    glfwSetMousePosCallback(mouse_position_callback)
    glfwSetMouseWheelCallback(mouse_wheel_callback)
    glfwSetKeyCallback(key_callback)
    glfwSetCharCallback(char_callback)


    print "Key repeat should be",
    if keyrepeat: print "enabled"
    else:         print "disabled"

    print "System keys should be",
    if systemkeys: print "enabled"
    else:          print "disabled"

    while (glfwGetWindowParam( GLFW_OPENED ) == GL_TRUE):
        glfwWaitEvents()
        glClear( GL_COLOR_BUFFER_BIT )
        glfwSwapBuffers()
    glfwTerminate()

