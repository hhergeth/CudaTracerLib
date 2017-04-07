cmake_minimum_required(VERSION 3.0)
project(CudaTracerLib)

set (CMAKE_CXX_STANDARD 14)

FIND_PACKAGE( Boost COMPONENTS system filesystem iostreams REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

# Specify include directories
set(QMATRIX_INCLUDE_DIR "qMatrix" CACHE FILEPATH "qMatrix")
include_directories(${QMATRIX_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_path(FREEIMAGE_INCLUDE_DIR FreeImage.h)
include_directories(${FREEIMAGE_INCLUDE_DIR})

find_package(CUDA REQUIRED)
# Pass options to NVCC
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-rdc=true")
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_61,code=sm_61
    )
set(CUDA_SEPARABLE_COMPILATION ON)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -m64")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -m64 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11")
SET(CMAKE_EXE_LINKER_FLAGS "-m64")
SET(CMAKE_MODULE_LINKER_FLAGS "-m64")
SET(CMAKE_SHARED_LINKER_FLAGS "-m64")

file(GLOB CTL_SRC_ROOT
       ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CTL_SRC
       ${PROJECT_SOURCE_DIR}/Base/*.cpp
       ${PROJECT_SOURCE_DIR}/Engine/*.cpp
       ${PROJECT_SOURCE_DIR}/Integrators/*.cpp
       ${PROJECT_SOURCE_DIR}/Kernel/*.cpp
       ${PROJECT_SOURCE_DIR}/Math/*.cpp
       ${PROJECT_SOURCE_DIR}/SceneTypes/*.cpp
)

file(GLOB CTL_CU_ROOT
       ${PROJECT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE CTL_CU
       ${PROJECT_SOURCE_DIR}/Base/*.cu
       ${PROJECT_SOURCE_DIR}/Engine/*.cu
       ${PROJECT_SOURCE_DIR}/Integrators/*.cu
       ${PROJECT_SOURCE_DIR}/Kernel/*.cu
       ${PROJECT_SOURCE_DIR}/Math/*.cu
       ${PROJECT_SOURCE_DIR}/SceneTypes/*.cu
)


set(LIB_NAME "CudaTracerLib")

cuda_add_library(${LIB_NAME} ${CTL_SRC} ${CTL_CU} ${CTL_SRC_ROOT} ${CTL_CU_ROOT})
target_link_libraries(${LIB_NAME} freeimage.so)
target_link_libraries(${LIB_NAME} ${Boost_LIBRARIES})
