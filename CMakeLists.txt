# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 3.8)
project (HELLO)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS_DEBUG "-g")
set (CMAKE_CXX_STANDARD 17)

# set(CMAKE_CXX_FLAGS_RELEASE "-g -O0 --std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=native --std=c++17")

###################################
# System specific options.
###################################
if(APPLE)
	ADD_COMPILE_OPTIONS(-framework Accelerate -DACCELERATE_NEW_LAPACK)
endif (APPLE)

IF (WIN32)
	ADD_COMPILE_OPTIONS("-Wall /ZI /DEBUG /Qxhost -O3" )
ENDIF (WIN32)


###################################
# Compiler specific Options.
###################################
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	
endif()

if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM"))
	set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -xHost")
endif()

###################################
# Status
###################################
message("System: ${CMAKE_SYSTEM_NAME}")
message("Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message("Flags: ${CMAKE_CXX_FLAGS_RELEASE}")

###################################
# Define executable
###################################
set(MY_EXEC_NAME "test-${CMAKE_CXX_COMPILER_ID}")
add_executable(${MY_EXEC_NAME} kernels/kernel-eigen.cpp kernels/kernel-sycl.cpp kernels/kernel-ug4.cpp demo.cpp)

###############################
# OpenMP 
###############################
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
	message(STATUS "** OpenMP!")
	add_compile_definitions(USE_OPENMP)
	target_link_libraries(${MY_EXEC_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif (OpenMP_FOUND)

###############################
# BLAS (optional).
###############################
# set(BLA_VENDOR Intel)

message(STATUS "Checking BLAS (select version using BLA_VENDOR=${BLA_VENDOR})...")
find_package(BLAS)

if (BLAS_FOUND)
   message(STATUS "** BLAS: ${BLAS_LIBRARIES}, ${BLAS_LINKER_FLAGS}")
   message(STATUS "Info: Using BLAS (Include: ${BLAS_INCLUDE_DIR}, Lib: ${BLAS_LIBRARIES})")	
   IF (APPLE)
   	  add_compile_definitions(USE_CBLAS)
   ELSE (APPLE)
  	  add_compile_definitions(USE_BLAS)
   ENDIF (APPLE)	
   
   target_link_libraries(${MY_EXEC_NAME} PUBLIC ${BLAS_LIBRARIES} BLAS::BLAS)  
else (BLAS_FOUND)
	message(STATUS "BLAS not found!")
endif (BLAS_FOUND)


###############################
# MKL (optional)
###############################
if (NOT BLAS_FOUND)
message(STATUS "Checking for Intel MKL...")
find_package(MKL)
if (MKL_FOUND)
	message(STATUS "...found!")
	add_compile_definitions(USE_MKL_BLAS)
	add_compile_options($<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
	target_include_directories (${MY_EXEC_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
	target_link_libraries(${MY_EXEC_NAME} PUBLIC $<LINK_ONLY:MKL::MKL>)
else (MKL_FOUND)

endif (MKL_FOUND)
message(STATUS "... not found!")
endif (NOT BLAS_FOUND)

###############################
# Eigen3 (optional)
###############################
find_package (Eigen3 3.3 NO_MODULE)
if(Eigen3_FOUND) 
 	message(STATUS "** Eigen3: ${Eigen3_FOUND}")
	add_compile_definitions(USE_EIGEN3)
	target_link_libraries(${MY_EXEC_NAME} PUBLIC Eigen3::Eigen)
endif(Eigen3_FOUND) 

###############################
# UG4 (optional)
###############################
if(DEFINED ENV{UG4_ROOT})
	message(STATUS "** UG4: $ENV{UG4_ROOT}")
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
 	message(STATUS "** Intel-SYCL: ${IntelSYCL_FOUND}")
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
# OpenCL (optional)
###############################
find_package(OpenCL)
message(STATUS "** OpenCL: ${OpenCL_FOUND}")
if(OpenCL_FOUND) 
	# target_link_libraries(vadd OpenCL::OpenCL)
	add_compile_definitions(USE_OPENCL)
	target_include_directories (${MY_EXEC_NAME} PUBLIC OpenCL::OpenCL)
endif() 

###############################
# Apple Metal (optional)
###############################
if(0) 
	# target_link_libraries(vadd OpenCL::OpenCL)
	add_compile_definitions(USE_METAL)
	add_library(metal kernels/metal/kernel-metal.cpp)
	include_directories (AFTER kernels/metal/metal-cpp)
endif() 

###############################
# Torch (optional)
###############################
find_package(Torch)
message(STATUS "** Torch: ${Torch_FOUND}")
if(Torch_FOUND) 
	message(STATUS "lib: ${TORCH_LIBRARY}")
	message(STATUS "inc: ${TORCH_INCLUDE_DIRS}")

	add_compile_definitions(USE_TORCH)

	# kernels/test with torch
	add_library(kerneltorch kernels/torch/kernel-torch.cpp)

	target_include_directories (kerneltorch PUBLIC ${TORCH_INCLUDE_DIRS})
	target_link_libraries(kerneltorch PUBLIC ${TORCH_LIBRARIES})
	# target_include_directories (${MY_EXEC_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
	target_link_libraries(${MY_EXEC_NAME} PUBLIC kerneltorch ${TORCH_LIBRARY} c10)

	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  		#set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
		#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
	endif()
endif() 


###############################
# Google benchmark (optional)
###############################
find_package(benchmark)
message(STATUS "** Google benchmark: ${benchmark_FOUND}")
if(benchmark_FOUND) 
	add_compile_definitions(-DUSE_GOOGLE_BENCHMARK)
endif(benchmark_FOUND) 



# if(OpenMP_CXX_FOUND)
target_compile_definitions(${MY_EXEC_NAME} PUBLIC ${HAS_BLAS} ${HAS_MKL_BLAS})
target_compile_options(${MY_EXEC_NAME} PUBLIC -flax-vector-conversions ${MyCompileOptions})
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	target_compile_options(${MY_EXEC_NAME} PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=0")
endif()

