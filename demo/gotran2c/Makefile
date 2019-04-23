
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

# 
demo.o: tentusscher_panfilov_2006_M_cell.h
tentusscher_panfilov_2006_M_cell.h: tentusscher_panfilov_2006_M_cell.ode
	gotran2c $< --solvers.explicit_euler.generate=1 --solvers.rush_larsen.generate=1 --code.body.use_enum=1 --output=$@

clean:
	$(RM) $(libfile)
	$(RM) $(executable)
	$(RM) $(wildcard *.o)
