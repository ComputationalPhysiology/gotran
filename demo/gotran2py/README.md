# Demo - gotran2py

In this demo we will show how you can generate a python module from an
ode-file. To illustrate the usage we will use the
`tentusscher_panfilov_2006_M_cell.ode` file.

To generate a python module type

```
gotran2py tentusscher_panfilov_2006_M_cell.ode
```

This will create a new file called
`tentusscher_panfilov_2006_M_cell.py` which contains the relevant
functions. In `demo.py` we illustrtee with the use of the `scipy`
odesolver how you can solve this ODE and plot the results. 

To run the demo type
```
python3 demo.py
```


A file called `results_python.png` will be save in this directory with
a plot of the membrane potential and the Kr current.


Note that by typing 
```
gotran2py --help
```

you can get all the options for codegeneration. 



