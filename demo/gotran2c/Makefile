
sources = demo.c
executable = demo
libfile = libdemo.so

CFLAGS = -Wall -O3 -ffast-math -march=native
LDLIBS = -lm

all: $(executable) $(libfile)
#all: $(libfile)
$(executable): demo.o

$(libfile): CFLAGS += -fPIC
$(libfile): $(sources)
	$(CC) $(CFLAGS) -shared -o $@ $^

clean:
	$(RM) $(libfile)
	$(RM) $(executable)
