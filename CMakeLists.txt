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

# this is a modified version from FindCUDA because under windows the maximum argument length of CMD.exe
# is only 8kb which is not enough for the CUDA link step
# therefore this version (only used for windows) converts absolute obj file paths to relative ones
# this is done to circumvent the issue since there is no solution
macro(CUDA_ADD_EXECUTABLE_EXT cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )
    
  # Remove all absolute paths and make the object paths relative to the binary directory
  string(REPLACE "${CMAKE_BINARY_DIR}/" "" cnv_cmp_targets "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")
    
  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${cnv_cmp_targets}")

  # Add the library.
  add_executable(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${cnv_cmp_targets}")

  target_link_libraries(${cuda_target} ${CUDA_LINK_LIBRARIES_KEYWORD}
    ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()

if(WIN32)
    CUDA_ADD_EXECUTABLE_EXT(${LIB_NAME} ${INC_FILES})
else()
    CUDA_ADD_EXECUTABLE(${LIB_NAME} ${INC_FILES})
endif()

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