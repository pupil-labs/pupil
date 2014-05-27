INC= -I/opt/local/include -I/usr/local/include

CFLAGS= -shared -fPIC -Wall 

LIBS = -framework IOKit -framework CoreFoundation  -L/usr/local/lib
OBJ = libuvcc.c

make_all: uvcc.so

uvcc.so: $(OBJ)
	-clang $(CFLAGS) -o $@ $^ $(INC) $(LIBS) # $^ = dependencies , $@=Target
	#mv *.so c/
# uvcc.o: $(OBJ)
# 	-clang $(CFLAGS) -c $^ $(INC) $(LIBS)

# uvcc.so: uvcc-wrappers.c uvcc.o
# 	-clang $(CFLAGS) -o $@ $^ $(INC) $(LIBS)  # $^ = dependencies , $@=Target
# 	#mv *.so c/

.PHONY: clean

clean:
	rm *.so