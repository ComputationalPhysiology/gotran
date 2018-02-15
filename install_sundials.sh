#!/bin/bash

INSTALL_PREFIX="~/local"
# SET THE BLAS DIRECTORY
BLAS_DIR="/usr"
BLAS_DIR="/usr/local"
BLAS_DIR="~/anaconda2"
# YOU can e.g install blas using "conda install -c conda-forge blas"

# SET THE LAPACK DIRECTORY
LAPACK_DIR="/usr"
LAPACK_DIR="/usr/local"
LAPACK_DIR="~/anaconda2"
# Install it with "conda install -c conda-forge lapack"


CURRENT_DIR=`pwd`
wget https://computation.llnl.gov/projects/sundials/download/sundials-2.6.0.tar.gz
tar -xvf sundials-2.6.0.tar.gz
cd sundials-2.6.0
export SUNDIALS_DIR=`pwd`"/sundials"
mkdir -p build
cd build

mkdir -p $SUNDIALS_DIR
cmake -DMPI_ENABLE=ON -DOPENMP_ENABLE=ON -DCMAKE_CFLAGS="-fPIC" -DEXAMPLES_INSTALL_PATH=$SUNDIALS_DIR/examples -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR ..
make
make install
cd $CURRENT_DIR
wget https://pypi.python.org/packages/4c/c0/19a54949817204313efff9f83f1e4a247edebed0a1cc5a317a95d3f374ae/Assimulo-2.9.zip
unzip Assimulo-2.9.zip
cd Assimulo-2.9

python setup.py install --sundials-home=$SUNDIALS_DIR --blas-home=$BLAS_DIR --lapack-home=$LAPACK_DIR --prefix=$INSTALL_PREFIX
