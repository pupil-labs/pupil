INC= -I/opt/local/include -I/usr/local/include

CFLAGS= -shared -fPIC -Wall

LIBS = -framework IOKit -framework CoreFoundation  -L/usr/local/lib
OBJ = libuvcc.c

make_all: uvcc.so

uvcc.so: $(OBJ)
	-gcc $(CFLAGS) -o $@ $^ $(INC) $(LIBS) # $^ = dependencies , $@=Target
	#mv *.so c/
# uvcc.o: $(OBJ)
# 	-gcc $(CFLAGS) -c $^ $(INC) $(LIBS)

# uvcc.so: uvcc-wrappers.c uvcc.o
# 	-gcc $(CFLAGS) -o $@ $^ $(INC) $(LIBS)  # $^ = dependencies , $@=Target
# 	#mv *.so c/

.PHONY: clean

clean:
	rm *.so