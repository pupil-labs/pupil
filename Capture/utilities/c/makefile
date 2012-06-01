INC=-I /usr/local/Cellar/libusb/1.0.8/include/libusb-1.0/

CFLAGS= -shared -fPIC -Wall 

LIBS = /usr/local/Cellar/libusb/1.0.8/lib/libusb-1.0.0.dylib

OBJ = lsusb.c

make_all: lsusb.so

lsusb.so: $(OBJ)  $(LIBS) 
	-gcc $(CFLAGS) -o $@ $^ $(INC) # $^ = dependencies , $@=Target
	#mv *.so c/
	
.PHONY: clean
	
clean:
	rm *.so