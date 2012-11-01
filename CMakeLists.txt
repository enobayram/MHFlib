CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  #SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

SET(PYTHON_SCRIPTS_DIR ${PROJECT_SOURCE_DIR}/MHFPython/scripts)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

FIND_PACKAGE(Boost 1.45.0 REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}")

ADD_EXECUTABLE(demo MHF/demo.cpp)

FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(Boost COMPONENTS python)

IF(Boost_PYTHON_FOUND AND PYTHONLIBS_FOUND)

  INCLUDE_DIRECTORIES("${PYTHON_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/MHF")

  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)

  ADD_LIBRARY(MHFPython SHARED MHFPython/GHList.cpp MHFPython/matrices.cpp 
                               MHFPython/mhd.cpp MHFPython/mhfpython.cpp 
                               MHFPython/table.cpp MHFPython/transformations.cpp
                               MHFPython/ukf.cpp MHFPython/visualization.cpp)

  TARGET_LINK_LIBRARIES(MHFPython ${Boost_LIBRARIES})
  SET_TARGET_PROPERTIES(MHFPython PROPERTIES PREFIX "")

  ADD_CUSTOM_COMMAND( TARGET MHFPython
                      COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_SCRIPTS_DIR}/demo.py ${CMAKE_BINARY_DIR}/
                      COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_SCRIPTS_DIR}/plotGaussians.py ${CMAKE_BINARY_DIR}/)
ENDIF() 

FIND_PACKAGE(Boost COMPONENTS program_options)

IF(Boost_PROGRAM_OPTIONS_FOUND)
  ADD_EXECUTABLE(generatetable generatetable/src/generatetable.cpp)
  TARGET_LINK_LIBRARIES(generatetable ${Boost_LIBRARIES})
ENDIF()




IF(NOT Boost_PYTHON_FOUND OR NOT PYTHONLIBS_FOUND )
  MESSAGE(WARNING "Boost Python library or Python not found, the python demo is skipped")
ENDIF()

IF(NOT Boost_PROGRAM_OPTIONS_FOUND)
  MESSAGE(WARNING "Boost Program Options library is not found, the table generator skipped")
ENDIF()
