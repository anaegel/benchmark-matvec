# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 3.12)
project (HELLO)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS_DEBUG "-g")
set (CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=native")
#set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=native --std=c++17")

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
option(ENABLE_EIGEN3 "Enable Eigen3 support" OFF)
if(ENABLE_EIGEN3)
	find_package (Eigen3 3.3 NO_MODULE)
	if(Eigen3_FOUND)
		message(STATUS "** Eigen3: ${Eigen3_FOUND}")
		add_compile_definitions(USE_EIGEN3)
		target_link_libraries(${MY_EXEC_NAME} PUBLIC Eigen3::Eigen)
	endif()
else()
	message(STATUS "Eigen3 support disabled (ENABLE_EIGEN3=OFF)")
endif()

###############################
# UG4 (optional)
###############################
option(ENABLE_UG4 "Enable UG4 support" OFF)
if(ENABLE_UG4)
	if(DEFINED ENV{UG4_ROOT})
		message(STATUS "** UG4: $ENV{UG4_ROOT}")
		add_compile_definitions(USE_UG4)
		target_include_directories (${MY_EXEC_NAME} PUBLIC 
			"$ENV{UG4_ROOT}/ugcore/ugbase" 
			"$ENV{UG4_ROOT}/externals/BoostForUG4")
	else()
		message(STATUS "UG4 not found.")
	endif()
else()
	message(STATUS "UG4 support disabled (ENABLE_UG4=OFF)")
endif()

###############################
# SYCL (experimental)
###############################
option(ENABLE_INTEL_SYCL "Enable Intel SYCL support" OFF)
if(ENABLE_INTEL_SYCL)
	find_package(IntelSYCL)
	if(IntelSYCL_FOUND)
		message(STATUS "** Intel-SYCL: ${IntelSYCL_FOUND}")
		add_compile_definitions(USE_SYCL)
		add_sycl_to_target(TARGET ${MY_EXEC_NAME})
	endif()
else()
	message(STATUS "Intel SYCL support disabled (ENABLE_INTEL_SYCL=OFF)")
endif()



option(ENABLE_ADAPTIVECPP "Enable AdaptiveCPP SYCL support" OFF)
if(ENABLE_ADAPTIVECPP)
	find_package(AdaptiveCPP)
	if(AdaptiveCPP_FOUND)
		message(STATUS "Found Adaptive-CPP-SYCL: ${AdaptiveCPP_FOUND}")
		add_compile_definitions(USE_SYCL)
		cmake_policy(SET CMP0005 NEW)
		add_sycl_to_target(TARGET ${MY_EXEC_NAME})
	endif()
else()
	message(STATUS "AdaptiveCPP support disabled (ENABLE_ADAPTIVECPP=OFF)")
endif()


###############################
# RAJA (optional)
###############################
option(ENABLE_RAJA "Enable RAJA support" OFF)
if(ENABLE_RAJA)
	find_package(RAJA REQUIRED)
	# blt_print_target_properties(TARGET RAJA)
	add_compile_definitions(USE_RAJA)
else()
	message(STATUS "RAJA support disabled (ENABLE_RAJA=OFF)")
endif()

###############################
# OpenCL (optional)
###############################
option(ENABLE_OPENCL "Enable OpenCL support" OFF)
if(ENABLE_OPENCL)
	find_package(OpenCL)
	message(STATUS "** OpenCL: ${OpenCL_FOUND}")
	if(OpenCL_FOUND)
		# target_link_libraries(vadd OpenCL::OpenCL)
		add_compile_definitions(USE_OPENCL)
		target_include_directories (${MY_EXEC_NAME} PUBLIC OpenCL::OpenCL)
	endif()
else()
	message(STATUS "OpenCL support disabled (ENABLE_OPENCL=OFF)")
endif()

###############################
# Apple Metal (optional)
###############################
option(ENABLE_APPLE_METAL "Enable Apple Metal support" OFF)
if(ENABLE_APPLE_METAL)
	add_compile_definitions(USE_METAL)
	add_library(metal kernels/metal/kernel-metal.cpp)
	include_directories (AFTER kernels/metal/metal-cpp)
else()
	message(STATUS "Apple Metal support disabled (ENABLE_APPLE_METAL=OFF)")
endif()

###############################
# Kokkos (optional)
###############################
option(ENABLE_KOKKOS "Enable build with Kokkos backend" ON)

if(ENABLE_KOKKOS)
	# Prefer a local submodule if present
	set(KOKKOS_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/kernels/kokkos/kokkos")
	if(EXISTS "${KOKKOS_SUBMODULE_DIR}/CMakeLists.txt")
		message(STATUS "Using Kokkos submodule at ${KOKKOS_SUBMODULE_DIR}")
		add_subdirectory(${KOKKOS_SUBMODULE_DIR} EXCLUDE_FROM_ALL)
	else()
		# Try find system-installed Kokkos first (prefer a released config)
		find_package(Kokkos 4.2 CONFIG QUIET)
		message(STATUS "** Kokkos found by find_package: ${Kokkos_FOUND}")

		if(NOT Kokkos_FOUND)
			message(STATUS "Kokkos not found via find_package; will try FetchContent fallback (this may clone the Kokkos repo).")
			include(FetchContent)

			# Minimal Kokkos FetchContent fallback. Assumption: it's acceptable to fetch/build
			# a serial-only Kokkos for local testing. If you want OpenMP/CUDA/HIP enablement,
			# set the corresponding Kokkos cache variables before configuring.
			set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
			set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
			set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)

			# NOTE: Choosing 'master' here as a conservative default; override by setting
			# Kokkos_GIT_TAG in your CMake cache if you prefer a specific release.
			FetchContent_Declare(
				kokkos
				GIT_REPOSITORY https://github.com/kokkos/kokkos.git
				GIT_TAG master
			)
			FetchContent_MakeAvailable(kokkos)
		endif()
	endif()

	# Accept either a proper find_package result (Kokkos_FOUND), that the
	# FetchContent build exposed the imported target Kokkos::kokkos or that the
	# submodule added the Kokkos target via add_subdirectory.
	if(Kokkos_FOUND OR TARGET Kokkos::kokkos)
		message(STATUS "** Kokkos status: available")
		add_compile_definitions(USE_KOKKOS)
		add_library(kernelkokkos kernels/kokkos/kernel-kokkos.cpp)
		target_link_libraries(kernelkokkos PRIVATE Kokkos::kokkos)
		target_compile_features(kernelkokkos PUBLIC cxx_std_17)
		target_link_libraries(${MY_EXEC_NAME} PRIVATE kernelkokkos Kokkos::kokkos)

	# Optional small test target that runs a Kokkos smoke test.
	add_executable(test-kokkos kernels/kokkos/kokkos_test.cpp)
	target_link_libraries(test-kokkos PRIVATE Kokkos::kokkos)
	target_compile_features(test-kokkos PUBLIC cxx_std_17)
	enable_testing()
	add_test(NAME test-kokkos COMMAND test-kokkos)
	# Set conservative timeout and ensure test runs in the build directory
	set_tests_properties(test-kokkos PROPERTIES TIMEOUT 30 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

	# Platform-specific optimization hints: on Apple Silicon enable -mcpu for
	# better vector instruction selection. These flags are added only to the
	# kokkos-related targets so they don't override user toolchain settings.
	if(APPLE)
		message(STATUS "Applying Apple SIMD compile options for Kokkos targets: -O3 -mcpu=apple-m1 -ffast-math")
		target_compile_options(kernelkokkos PRIVATE -O3 -mcpu=apple-m1 -ffast-math)
		target_compile_options(test-kokkos PRIVATE -O3 -mcpu=apple-m1 -ffast-math)
	endif(APPLE)
	else()
		message(STATUS "Kokkos still not available after fallback; building without Kokkos kernels.")
	endif()
else()
	message(STATUS "Kokkos integration disabled by ENABLE_KOKKOS=OFF")
endif(ENABLE_KOKKOS)

###############################
# Torch (optional)
###############################
option(ENABLE_TORCH "Enable build with Torch backend" OFF)

if(ENABLE_TORCH)
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
endif(Torch_FOUND) 
endif(ENABLE_TORCH)


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

