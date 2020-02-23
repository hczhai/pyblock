CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

FIND_PROGRAM(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
FIND_PROGRAM(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

SET(PROJECT_NAME newmat)
PROJECT(${PROJECT_NAME} VERSION 1.0)

ADD_LIBRARY(${PROJECT_NAME} STATIC newmat1.cpp newmat2.cpp newmat3.cpp newmat4.cpp newmat5.cpp newmat6.cpp
    newmat7.cpp newmat8.cpp newmatex.cpp bandmat.cpp submat.cpp myexcept.cpp cholesky.cpp evalue.cpp
    fft.cpp hholder.cpp jacobi.cpp newfft.cpp sort.cpp svd.cpp newmatrm.cpp newmat9.cpp)

TARGET_COMPILE_OPTIONS(${PROJECT_NAME} BEFORE PRIVATE -fPIC -O2 -Wall -Wno-misleading-indentation -Wno-terminate)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE .)
