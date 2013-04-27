override CFLAGS += -shared -fPIC -Wall -O3

OBJ = methods.c

make_all: methods.so

methods.so: $(OBJ)
	-gcc $(CFLAGS) -o $@ $^

.PHONY: clean

clean:
	rm *.so