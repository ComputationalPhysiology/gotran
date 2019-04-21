# Demo - gotran2c

In this demo we will show how you can generate a python module from an
ode-file. To illustrate the usage we will use the
`tentusscher_panfilov_2006_M_cell.ode` file.

To generate a python module type

```
gotran2c tentusscher_panfilov_2006_M_cell.ode --solvers.explicit_euler.generate=1 --solvers.rush_larsen.generate=1 --code.body.use_enum=1
```

The first two aguments states that we want to generate schemes for
[exclicit euler](https://en.wikipedia.org/wiki/Euler_method) and for
[rush larsen](https://arxiv.org/abs/1712.02260). The final argument states that we want to use enum
based indexing which makes it easier to keep track of which states and parameters belong to which index.

This will create a new file called
`tentusscher_panfilov_2006_M_cell.h` which contains the relevant
functions. This is a header file that we will include in `demo.c` file
where we will illustrate how you can use the two schemes to solve the
ode. You can compile the c-code into a binary by typing 

```
make
```

This will copmile the code into a binary and also crate a library file
that we will use later.

To compare the two schemes you can try to execute the binary by typing

```
./demo
```

The output on my computer is the following

```
$ ./demo
Scheme: Forward Euler
Computed 1000000 time steps in 0.342376 s. Time steps per second: 2.92077e+06

Scheme: Rush Larsen (exp integrator on all gates)
Computed 1000000 time steps in 0.643319 s. Time steps per second: 1.55444e+06
```

Now to actaully work with this code and plot the results can be
difficult in pure C. Therefore you will find a `demo.py` file which is
a python file that imports the library file, solves the ODE and plots
the membrane potential. 

Thanks to Kristian Gregorius Hustad for help with wriing this demo.


