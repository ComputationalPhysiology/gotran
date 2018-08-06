from python:3.6

MAINTAINER Henrik Finsberg <henriknf@simula.no>

WORKDIR /app

# Install some tools
RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y swig && \
    apt-get install unzip && \
    apt-get install -y gfortran && \
    apt-get install -y python-tk && \
    apt-get install -y python-dev && \
    apt-get install -y python3-dev

# Use "bash" as replacement for	"sh"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Set up a virtual enviroment
RUN pip install --upgrade virtualenv
# Create python 3.6 virtual env
RUN virtualenv --python python3.6 venv
# Make sure to update LD_LIBRARY_PATH so that we find sundial lib files
RUN echo 'export LD_LIBRARY_PATH=/app/venv/lib/:$LD_LIBRARY_PATH' >> /app/venv/bin/activate
# Repeat for python 2.7 
RUN virtualenv --python python2.7 py27venv
RUN echo 'export LD_LIBRARY_PATH=/app/py27venv/lib/:$LD_LIBRARY_PATH' >> /app/py27venv/bin/activate

# Install sundials
COPY install_sundials.sh /app

# Python 2.7
RUN source /app/py27venv/bin/activate && \
    pip install numpy cython matplotlib scipy && \
    /bin/bash install_sundials.sh /app/py27venv 1 && \
    cd Assimulo-2.9 && \
    python setup.py install --sundials-home=/app/py27venv --prefix=/app/py27venv && \
    cd ..

# Python 3.6
RUN source /app/venv/bin/activate && \
    pip install numpy cython matplotlib scipy && \
    /bin/bash install_sundials.sh /app/venv 1 && \
    cd Assimulo-2.9 && \
    python setup.py install --sundials-home=/app/venv --prefix=/app/venv && \
    cd ..

ENTRYPOINT ["/bin/bash"]
