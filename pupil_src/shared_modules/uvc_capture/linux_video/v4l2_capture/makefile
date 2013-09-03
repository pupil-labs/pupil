override CFLAGS += -shared -fPIC -O2 -Wall `pkg-config --cflags --libs libv4l2` #is this requried?
OBJ = capture.c

make_all: capture.so

capture.so: $(OBJ)
	-gcc $(CFLAGS) -o $@ $^ -l v4l2  -I/usr/local/include

.PHONY: clean

clean:
	rm *.so