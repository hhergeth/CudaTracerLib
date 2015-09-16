# Introduction
This library, for simplicity called CudaTracerLib, is a CUDA based ray tracer library implementing standard rendering algorithms like

- Path Tracing
- Photon/Light Tracing
- Photon Mapping (Final Gathering)
- Volumetric(Beam x Beam) Adaptive Progressive Photon Mapping
- Bidirectional Path Tracing
- Vertex Connection and Merging

This library provides the possibility of using these algorithms for offline rendering but also for "realtime" rendering to an Open GL/D3D buffer.

Please keep in mind that this is mostly reseach code with the ultimate goal of developing new or improving old rendering algorithms. Therefore there is a lot of code in branches which are currently not used. While this might clutter the code, it is neccessary when trying new ideas.


# Installation
There are some dependecies

- Boost, used for all platform independent code
- FreeImage, used for loading and saving images
- qMatrixLib, a very small and lightweight, header only library for matrix algebra. You can get it here https://github.com/galrath434/qMatrixLib

After downloading the code, specify the appropriate C++ Include directories in the Visual Studio project file. The dependency .lib files are only needed when linking against the output .lib.


# Implementation
Throughout the library the design goal was to be able to use as much code as possible on the GPU as well as on the CPU. This was made possible by reimplementing some code which might seem counterintuitive at first.

The general layout of the source code is based on the following directories

- ROOT : CUDA memory manager, base classes for pseudo virtual classes in CUDA
- Math : Linear algebra math classes as well as sampling functions and function integrators
- Base : General purpose classes like timing, fixed size strings, high performance file streams and random number generators
- Engine : BSDFs, Emitters, Sensors, Image filters, Textures and all foundation a rendering algorithm needs.
- Kernel : Ray tracing operations, buffer management on CPU and GPU
- Integrators : Rendering algorithms listed above, some ported to "Wavefront Path Tracing" for efficiency on the GPU

The library supports many different material models as well as a multitude of light and camera types. For efficiency a two level BVH hierarchy is used. The first level BVH is used for objects in the scene and the second for triangles in each object.

While this is currently a Windows/Visual Studio only library, there are no real dependecies on Windows (there are probably some in the code but not many and by using Boost these should be easy to fix). The main challange with a linux port is the project file. The problem here is CUDA and specifying the appropriate flags for linking (with device relocatable code).


# Acknowledgements
I would like to thank Wenzel Jakob for allowing me to use a lot of his work from Mitsuba - http://www.mitsuba-renderer.org. This includes the general interfaces and implementation of the `BSDF`, `Emitter`, `Sensor` classes. Furthermore I used his `MicrofacetDistribution` and `RoughTransmittance` classes as well as the design of the `SamplingRecord` classes.

Timo Aila and Samuli Laine are to be thanked for their research on BVH ray traversal on CUDA GPUs in "Understanding the Efficiency of Ray Traversal on GPUs". I used slight modifications of their code for the BVH computation as well as the traversal.


# Limitations
Currently there are no Metropolis MCMC algorithms implemented. The reasons are two fold, it is necessary to have some sort of Path/Path-Vertex classes present which makes memory managment mandatory on the GPU. Also it is difficult to design a well performing sampler for MCMC which spreads the work on the GPU.


# Issuses
- `size_t` vs `unsigned int` In some places `size_t` is used while in others not. This mistake was made when this project started but is actually harder to fix because of performance considerations for the GPU.
- Another design error made very early in the development, was to not use namespaces. While this is fine for smaller projects this is certainly incorrect here.
- The GPU optimized ray traversal algorithms are not working correctly, this issue appears due to the complicated nested BVH layout. Therefore the standard ray traversal is used which takes no advantage of the CUDA architecture with lanes and warp voting.


# License
This library is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License Version 3 as published by the Free Software Foundation.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.