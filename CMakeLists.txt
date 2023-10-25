# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 3.8)
project (HELLO)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set(CMAKE_CXX_FLAGS "-Wall /ZI /DEBUG /Qxhost -O3" )
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set (CMAKE_CXX_STANDARD 17)

# set(CMAKE_CXX_FLAGS_RELEASE "-g -O0 --std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=native -framework Accelerate" )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	
endif()

if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM"))
	set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -xHost")
endif()


message("Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message("Flags: ${CMAKE_CXX_FLAGS_RELEASE}")


set(MY_EXEC_NAME "test-${CMAKE_CXX_COMPILER_ID}")
add_executable(${MY_EXEC_NAME} demo.cpp)

###############################
# OpenMP (optional)
###############################
find_package(OpenMP)

find_package(MKL)
if (MKL_FOUND)
	add_compile_definitions(USE_MKL_BLAS)
endif (MKL_FOUND)


###############################
# BLAS
###############################
# set(BLA_VENDOR Intel)
find_package(BLAS)
message(STATUS "Found BLAS: ${BLAS_FOUND}")



###############################
# Eigen3 (optional)
###############################
find_package (Eigen3 3.3 NO_MODULE)
if(Eigen3_FOUND) 
 	message(STATUS "Found Eigen3: ${Eigen3_FOUND}")
	add_compile_definitions(USE_EIGEN3)
endif(Eigen3_FOUND) 

###############################
# UG4 (optional)
###############################
if(DEFINED ENV{UG4_ROOT})
	message(STATUS "Found UG4: $ENV{UG4_ROOT}")
	add_compile_definitions(USE_UG4)
	target_include_directories (${MY_EXEC_NAME} PUBLIC 
	"$ENV{UG4_ROOT}/ugcore/ugbase" 
	"$ENV{UG4_ROOT}/externals/BoostForUG4")
else(DEFINED ENV{UG4_ROOT})
	message(STATUS "UG4 not found.")
endif(DEFINED ENV{UG4_ROOT})

###############################
# SYCL (experimental)
###############################
find_package(IntelSYCL)
if(IntelSYCL_FOUND) 
 	message(STATUS "Found Intel-SYCL: ${IntelSYCL_FOUND}")
	add_compile_definitions(USE_SYCL)
	add_sycl_to_target(TARGET ${MY_EXEC_NAME})
endif(IntelSYCL_FOUND) 


find_package(AdaptiveCPP)
if(AdaptiveCPP_FOUND) 
 	message(STATUS "Found Adaptive-CPP-SYCL: ${AdaptiveCPP_FOUND}")
	add_compile_definitions(USE_SYCL)
	cmake_policy(SET CMP0005 NEW)
	add_sycl_to_target(TARGET ${MY_EXEC_NAME})
endif(AdaptiveCPP_FOUND) 


if (DEFINED RAJA_DIR)
  find_package(RAJA REQUIRED)
  # blt_print_target_properties(TARGET RAJA)
  add_compile_definitions(USE_RAJA)
endif ()


###############################
# Google benchmark (optional)
###############################
find_package(benchmark)
message(STATUS "Found Google benchmark: ${benchmark_FOUND}")
if(benchmark_FOUND) 
 	
	add_compile_definitions(-DUSE_GOOGLE_BENCHMARK)
endif(benchmark_FOUND) 






###############################
# BLAS
###############################

if (BLAS_FOUND)
   set(HAS_BLAS "USE_BLAS")
   # set(HAS_MKL_BLAS "USE_MKL_BLAS")
   message(STATUS "Found BLAS: ${BLAS_LIBRARIES}, ${BLAS_LINKER_FLAGS}")
   message(STATUS "Info: Using BLAS (Include: ${BLAS_INCLUDE_DIR}, Lib: ${BLAS_LIBRARIES})")	
endif (BLAS_FOUND)




	
#  link_directories("/Users/anaegel/Software/ug4-git/lib")



if(OpenMP_CXX_FOUND)
	target_compile_definitions(${MY_EXEC_NAME} PUBLIC ${HAS_BLAS} ${HAS_MKL_BLAS})
	target_compile_options(${MY_EXEC_NAME} PUBLIC -flax-vector-conversions)
	target_link_libraries(${MY_EXEC_NAME} 
		PUBLIC 
		OpenMP::OpenMP_CXX 
		${BLAS_LIBRARIES} 
		Eigen3::Eigen 
		#	BLAS::BLAS
	)
		# ug4) 
	
endif()

