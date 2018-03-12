cmake_minimum_required(VERSION 3.0)
project(CudaTracerLib)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set (CMAKE_CXX_STANDARD 17)

# so that better include paths are possible
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add external dependencies
include_directories(ext)
include_directories("ext/ext-boost")
add_subdirectory(ext/FreeImage)
add_subdirectory(ext/pugixml)
add_subdirectory(ext/miniz)

# hide the external dependencies in visual studio
set_property(TARGET freeimage jpeg openexr openjpeg rawlite tiff zlib png mng PROPERTY FOLDER "dependencies")
set_property(TARGET pugixml PROPERTY FOLDER "dependencies")
set_property(TARGET miniz example1 example2 example3 example4 example5 example6 PROPERTY FOLDER "dependencies")

# Pass options to NVCC
find_package(CUDA REQUIRED)
# Cmake automatically links with CUBLAS which we don't want
SET(CUDA_cublas_device_LIBRARY "" CACHE PATH "ABC" FORCE)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION ON)

if(UNIX)
    # CUDA needs PIC so enable it on all relevant targets and use the CMAKE_LANG_FLAGS to pass it to the CUDA compiler
    target_compile_options(miniz PRIVATE -fPIC)
    target_compile_options(pugixml PRIVATE -fPIC)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# openexr is still using binary_function which has been deprecated in c++17
target_compile_definitions(openexr PRIVATE _HAS_AUTO_PTR_ETC=1)

# find all source files
function(add_files root rec)
	file(
		${rec} _source_list 
		LIST_DIRECTORIES false
		"${root}*.cpp*"
		"${root}*.h*"
		"${root}*.cu*"
	)
	set(INC_FILES ${INC_FILES} ${_source_list} PARENT_SCOPE)
endfunction(add_files)

add_files("" GLOB)
add_files("Base/" GLOB_RECURSE)
add_files("Engine/" GLOB_RECURSE)
add_files("Integrators/" GLOB_RECURSE)
add_files("Kernel/" GLOB_RECURSE)
add_files("Math/" GLOB_RECURSE)
add_files("SceneTypes/" GLOB_RECURSE)

# create target
set(LIB_NAME "CudaTracerLib")

set(BUILD_LIBRARY true CACHE BOOL "Whether to build a library or an executable")

if (BUILD_LIBRARY)
    cuda_add_library(${LIB_NAME} ${INC_FILES} SHARED)
else ()
    CUDA_ADD_EXECUTABLE(${LIB_NAME} ${INC_FILES})
endif ()

# add filters for visual studio projects
foreach(_source IN ITEMS ${INC_FILES})
    get_filename_component(_source_path "${_source}" PATH)
    file(RELATIVE_PATH _source_path_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()

# link with all the previously defined external dependencies
target_link_libraries(${LIB_NAME} freeimage jpeg openexr openjpeg rawlite tiff zlib png mng)
target_link_libraries(${LIB_NAME} pugixml)
target_link_libraries(${LIB_NAME} miniz)
