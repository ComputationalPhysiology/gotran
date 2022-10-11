MODEL=tentusscher_panfilov_2006_M_cell

all: $(MODEL).cl $(MODEL).py

$(MODEL).cl: $(MODEL).ode
	python -m gotran gotran2opencl $< --solvers.explicit_euler.generate=1 --solvers.explicit_euler.function_name=FE --solvers.generalized_rush_larsen.generate=1 --solvers.generalized_rush_larsen.function_name=GRL1 --code.body.use_enum=1 --output=$@

$(MODEL).py: $(MODEL).ode
	gotran2py $<

clean:
	$(RM) $(MODEL).cl
	$(RM) $(MODEL).py
	$(RM) -r __pycache__
