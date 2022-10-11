
sources = demo.c
executable = demo
libfile = libdemo.so

CFLAGS = -Wall -O3 -ffast-math -march=native
LDLIBS = -lm

gotran_generator = python -m gotran gotran2c

all: $(executable) $(libfile)
#all: $(libfile)
$(executable): demo.o

$(libfile): CFLAGS += -fPIC
$(libfile): $(sources) tentusscher_panfilov_2006_M_cell.h
	$(CC) $(CFLAGS) -shared -o $@ $<

#
demo.o: tentusscher_panfilov_2006_M_cell.h
tentusscher_panfilov_2006_M_cell.h: tentusscher_panfilov_2006_M_cell.ode
	$(gotran_generator) $< --solvers.explicit_euler.generate=1 --solvers.rush_larsen.generate=1 --code.body.use_enum=1 --output=$@

clean:
	$(RM) $(libfile)
	$(RM) $(executable)
	$(RM) $(wildcard *.o)
