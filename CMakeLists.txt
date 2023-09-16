# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 3.8)
project (HELLO)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

message("${CMAKE_CXX_COMPILER_ID}")
# set(CMAKE_CXX_FLAGS_RELEASE "-g -O0 --std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 --std=c++14 -march=native -framework Accelerate" )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -xHost")
endif()





# OpenMP
find_package(OpenMP)


# set(BLA_VENDOR Intel)
find_package(BLAS)

# Eigen3
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


# Google benchmark
find_package(benchmark REQUIRED)





set(MY_EXEC_NAME "test-${CMAKE_CXX_COMPILER_ID}")
add_executable(${MY_EXEC_NAME} demo.cpp)


if (BLAS_FOUND)
   set(HAS_BLAS "USE_BLAS")
   # set(HAS_MKL_BLAS "USE_MKL_BLAS")
   message(STATUS "Found BLAS: ${BLAS_LIBRARIES}, ${BLAS_LINKER_FLAGS}")
   message(STATUS "Info: Using BLAS (Include: ${BLAS_INCLUDE_DIR}, Lib: ${BLAS_LIBRARIES})")	
   # include_directory ()	
endif (BLAS_FOUND)


if(OpenMP_CXX_FOUND)
	target_compile_definitions(${MY_EXEC_NAME} PUBLIC ${HAS_BLAS} ${HAS_MKL_BLAS})
	target_compile_options(${MY_EXEC_NAME} PUBLIC -flax-vector-conversions)
	target_link_libraries(${MY_EXEC_NAME} PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} Eigen3::Eigen BLAS::BLAS) 
endif()

