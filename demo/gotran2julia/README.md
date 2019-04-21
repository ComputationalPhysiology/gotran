# Demo - gotran2julia

In this demo we will show how you can generate julia code to solve
your ODE. To illustrate the usage we will use the
`tentusscher_panfilov_2006_M_cell.ode` file.
[Julia](https://docs.julialang.org/en/v1/)

To generate the relevant julia 

```
gotran2julia tentusscher_panfilov_2006_M_cell.ode
```

This will create a new file called
`tentusscher_panfilov_2006_M_cell.jl` which contains the relevant
functions. In `demo.jl` we illustrtee with the use a package called
[DifferentialEqautions](http://docs.juliadiffeq.org/latest/) to solve
the ODE.


To run the demo type
```
julia demo.jl
```


Note that by typing 
```
gotran2py --help
```

you can get all the options for codegeneration. 



