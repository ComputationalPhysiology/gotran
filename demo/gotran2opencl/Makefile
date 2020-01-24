all: tentusscher_panfilov_2006_M_cell.cl tentusscher_panfilov_2006_M_cell.py

tentusscher_panfilov_2006_M_cell.cl: tentusscher_panfilov_2006_M_cell.ode
	gotran2opencl $< --solvers.explicit_euler.generate=1 --solvers.explicit_euler.function_name=FE --solvers.generalized_rush_larsen.generate=1 --solvers.generalized_rush_larsen.function_name=GRL1 --code.body.use_enum=1 --output=$@

tentusscher_panfilov_2006_M_cell.py: tentusscher_panfilov_2006_M_cell.ode
	gotran2py $<
