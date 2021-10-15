# Install


## Virtual environment

It is recommended you install gotran and all its dependencies
in a virtual environment, since gotran depends on some specific version
of some package.

Install virtualenv

```
pip install virtualenv --user
```

Create a virtual environment

```
virtualenv --python python3.6 venv
```
(swich to python2.7 if you prefer python 2)

Now activate the virtual environment

```
source venv/bin/activate
```
You are now ready to install the packages

## Install the dependencies

To install the requrements simply do
```
pip install -r requirements.txt
```

## Install gotran

To install gotran do

```
pip install . --upgrade
```

## Install sundials solver
If you want to use [sundials solvers](<https://computation.llnl.gov/projects/sundials>)
to solve the ODEs you can run the bash script

```
bash install_sundials.sh
```
If you want to install sundials in another location than the virtual environment you can parse the path as an argument
```
bash install_sundials.sh /path/to/install/directory
```
This will install the sundial solver and a python wrapping called [Assimulo](https://pypi.org/project/Assimulo/).


## Installation issues
If you have installation issues it there is a detailed installatin instructions in the [Dockerfile](Dockerfile). This docker image can also be found at [docker hub](https://hub.docker.com) - `finsberg/py27py36-sundials`
