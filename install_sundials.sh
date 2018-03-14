#!/bin/bash

# First of all you need CMAKE: "conda install -c conda-forge cmake"

# INSTALL_PREFIX="/usit/abel/u1/henriknf/local"
# INSTALL_PREFIX="/home/finsberg/local"
INSTALL_PREFIX="/usr/local"
# SET THE BLAS DIRECTORY
# BLAS_DIR="/usr/lib/libblas"
# BLAS_DIR="/usr/local"
# BLAS_DIR="/usit/abel/u1/henriknf/anaconda2"
# YOU can e.g install blas using "conda install -c conda-forge blas"

# SET THE LAPACK DIRECTORY
# LAPACK_DIR="/usr/lib"
# LAPACK_DIR="/usr/local"
# LAPACK_DIR="/usit/abel/u1/henriknf/anaconda2"
# Install it with "conda install -c conda-forge lapack"


CURRENT_DIR=`pwd`

#Get superlu
# SUPERLU_VERSION=4.1
# wget http://crd-legacy.lbl.gov/~xiaoye/SuperLU/superlu_$SUPERLU_VERSION.tar.gz
# tar -xvf superlu_$SUPERLU_VERSION.tar.gz
# rm superlu_$SUPERLU_VERSION.tar.gz
# SUPERLU=$CURRENT_DIR/SuperLU_$SUPERLU_VERSION
# cd $SUPERLU
# cp ../superlu_make.inc make.inc
# # edit the make.inc if next step is failing
# make blaslib
# make
# SUPERLU_INCLUDE_DIR=$SUPERLU/SRC
# SUPERLU_LIBRARY_DIR=$SUPERLU/lib



# cd $CURRENT_DIR

# GET SUPERLU_MT
SUPERLUMT_VERSION=2.4
wget http://crd-legacy.lbl.gov/~xiaoye/SuperLU/superlu_mt_$SUPERLUMT_VERSION.tar.gz

tar -xvf superlu_mt_$SUPERLUMT_VERSION.tar.gz

rm superlu_mt_$SUPERLUMT_VERSION.tar.gz

SUPERLU_MT=$CURRENT_DIR/SuperLU_MT_$SUPERLUMT_VERSION

cd $SUPERLU_MT
cp MAKE_INC/make.linux.openmp make.inc
# edit the make.inc if next step is failing
make blaslib
make
SUPERLUMT_INCLUDE_DIR=$SUPERLU_MT/SRC
SUPERLUMT_LIBRARY_DIR=$SUPERLU_MT/lib
SUPERLUMT_THREAD_TYPE="OPEN_MP"

#BLAS_DIR=$SUPERLUMT_LIBRARY_DIR

cd $CURRENT_DIR

wget https://computation.llnl.gov/projects/sundials/download/sundials-2.6.0.tar.gz
tar -xvf sundials-2.6.0.tar.gz
rm sundials-2.6.0.tar.gz
cd sundials-2.6.0
export SUNDIALS_DIR=$INSTALL_PREFIX
mkdir -p build
cd build

mkdir -p $SUNDIALS_DIR
# cmake -DCMAKE_CFLAGS="-fPIC" -DEXAMPLES_INSTALL_PATH=$SUNDIALS_DIR/examples -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR ..
cmake -DSUPERLUMT_ENABLE=ON -DSUPERLUMT_INCLUDE_DIR=$SUPERLUMT_INCLUDE_DIR -DSUPERLUMT_LIBRARY_DIR=$SUPERLUMT_LIBRARY_DIR -DSUPERLUMT_THREAD_TYPE=$SUPERLUMT_THREAD_TYPE -DMPI_ENABLE=ON -DEXAMPLES_ENABLE=OFF  -DOPENMP_ENABLE=ON -DCMAKE_CFLAGS="-g -O2 -fPIC" -DEXAMPLES_INSTALL_PATH=$SUNDIALS_DIR/examples -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR ..
make
make install
cd $CURRENT_DIR
wget https://pypi.python.org/packages/4c/c0/19a54949817204313efff9f83f1e4a247edebed0a1cc5a317a95d3f374ae/Assimulo-2.9.zip
unzip Assimulo-2.9.zip
rm Assimulo-2.9.zip
cd Assimulo-2.9

# python setup.py install --sundials-home=$SUNDIALS_DIR --blas-home=$BLAS_DIR --lapack-home=$LAPACK_DIR --superlu-home=$SUPERLU --prefix=$INSTALL_PREFIX --sundials-with-superlu 1

python setup.py install --sundials-home=$SUNDIALS_DIR --prefix=$INSTALL_PREFIX
