
NOTES
=====

* docker stuff

.. code:: bash

    docker build -t boostpy:latest .
    docker run -it --rm -v $PWD:/work/code boostpy
    docker commit <container id by `docker ps`> boostpy:latest

* manually build on mac

.. code:: bash

    # on mac
    g++-9 -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` \
        -undefined dynamic_lookup \
        ../src/bar.cpp -o bar`python3-config --extension-suffix`
    # on linux
    g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` \
        ../src/bar.cpp -o bar`python3-config --extension-suffix`

* BLOCK MPI build on ubuntu

.. code:: bash

    cd build
    rm CMakeCache.txt; cmake .. -DMPI=ON
    make VERBOSE=1 -j 4
    mpirun --allow-run-as-root -n 2 ./cblock

* BLOCK python librarybuild on ubuntu

.. code:: bash

    cd build
    rm CMakeCache.txt; cmake .. -DBUILD_LIB=ON
    make VERBOSE=1 -j 4

* Calculate lines:
  
.. code:: bash

    for i in `ls *.C *.h include/*.C include/*.h`; do wc -l $i | awk '{print $1}' >> AA ; done
    for i in `find . -name '*.py' -not -path '*/\.*'`; do wc -l $i | awk '{print $1}' >> AA ; done
    cat AA | awk '{n+=$1}; END {print n}'

* Install using ``pip3``

.. code:: bash

    pip3 install . -v
    pip3 uninstall .

* API docs generation

.. code:: bash

    cd docs
    sphinx-quickstart .
    sphinx-apidoc -o ./source ../build-cmake
    
