#!/bin/bash

# First of all you need CMAKE: "conda install -c conda-forge cmake"
if [[ $1 == "" ]]; then
    INSTALL_PREFIX=`pwd`/venv;
else
    INSTALL_PREFIX=$1;
fi

CURRENT_DIR=`pwd`
curl -LO 'https://computation.llnl.gov/projects/sundials/download/sundials-2.6.0.tar.gz'
tar -xvf sundials-2.6.0.tar.gz
rm sundials-2.6.0.tar.gz
cd sundials-2.6.0
export SUNDIALS_DIR=$INSTALL_PREFIX
mkdir -p build
cd build

mkdir -p $SUNDIALS_DIR

cmake -DEXAMPLES_ENABLE=OFF -DCMAKE_CFLAGS="-fPIC" -DEXAMPLES_INSTALL_PATH=$SUNDIALS_DIR/examples -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR ..
make
make install

cd $CURRENT_DIR
curl -LO 'https://files.pythonhosted.org/packages/4c/c0/19a54949817204313efff9f83f1e4a247edebed0a1cc5a317a95d3f374ae/Assimulo-2.9.zip'
unzip -o Assimulo-2.9.zip
rm Assimulo-2.9.zip

if [[ $2 == "" ]]; then
    cd Assimulo-2.9

    # python setup.py install --sundials-home=$SUNDIALS_DIR --blas-home=$BLAS_DIR --lapack-home=$LAPACK_DIR --superlu-home=$SUPERLU --prefix=$INSTALL_PREFIX --sundials-with-superlu 1
    python setup.py install --sundials-home=$SUNDIALS_DIR --prefix=$INSTALL_PREFIX
