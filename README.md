<p align="center">
<a href='https://github.com/hhergeth/CudaTracerLib/wiki/Example-Renderings' style='font-size:2.2em'>Examples</span></a> <br>
<a href="https://github.com/hhergeth/CudaTracerLib/wiki/Example-Renderings">
<img src="http://hhergeth.markab.uberspace.de/Git-Wiki-Files/thumbnails/loadBox_20150716_080109_PPPM.jpg" style="width:880px;">
</a>
</p>

# CudaTracerLib

### Introduction
This library, for simplicity called CudaTracerLib, is a CUDA based ray tracer library implementing standard rendering algorithms like Path Tracing, Progressive Photon Mapping, Bidirectional Path Tracing and many more. It can be used in an offline render mode where only one frame is computed. The other option is to compute the frames directly into an Open GL/D3D buffer enabling "real-time" ray traced images.

This project started first and foremost as a learning experience. Secondly it is a study of a GPU implementation of a research oriented ray tracing framework in which new algorithms could be developed. The focus on the GPU was chosen due to the fact that commonly a significant amount of time is spent waiting for rendering results while all the power a GPU has to offer is left unused. Thus the goal is to make this power available without having to focus too much on low level optimizations.


### Building from Source
There are only four dependencies:

- Cuda 7.5
- Boost, used for filesystem(compiled library!) and string functions
- FreeImage, used for loading and saving images. There are CMake versions available.
- qMatrixLib, a tiny header only matrix library, available here https://github.com/hhergeth/qMatrixLib

For Windows users there is a Visual Studio 2013 project file available where only the include and library paths to Boost, FreeImage and qMatrixLib have to be specified.
The CMake file can be used for all other platforms. Assuming Boost is installed on the machine, only the FreeImage and qMatrixLib path have to be specified.

The microfacet and spectral probability distribution files from Mitsuba/PBRT are also necessary. They can be obtained from the Mitsuba build or [here from the repository](https://www.mitsuba-renderer.org/repos/mitsuba/files/c7aac473729a3342dba801717419d8f630fe7560/data). Only the _ior_ and _microfacet_ folder are required.

### Acknowledgements
I would like to thank Wenzel Jakob for allowing me to use a lot of his work from Mitsuba - http://www.mitsuba-renderer.org. This includes the general interfaces and implementation of the `BSDF`, `Emitter`, `Sensor` classes. Furthermore I have used his `MicrofacetDistribution` and `RoughTransmittance` classes as well as the design of the `SamplingRecord` classes.

Thanks to Timo Aila and Samuli Laine for their research on BVH ray traversal on CUDA GPUs in *Understanding the Efficiency of Ray Traversal on GPUs*. I have used slight modifications of their code for the BVH computation as well as the traversal.


### Comparison to Mitsuba and Brigade
Mitsuba (as PBRT before) is probably the most commonly known realistic renderer geared towards research application.  It can be easily extended with nearly any custom integrator, and especially the Path classes are of great convenience for Metropolis Light Transport algorithms. This library has no intention of competing with Mitsuba in this research area. It is much harder to do memory management on the device and therefore no Metropolis Light Transport algorithm is implemented here. Mitsuba is also easier to debug and simply has the larger and better documented codebase.

On the other end of the spectrum is Brigade, the hope of true real-time ray tracing. Without having any insight into the code, I assume their focus lies more on the optimization aspect, not on a general approach to realistic image synthesis. For further discussion of the real-time aspects of this library please see the appropriate wiki page.

This library tries to place itself somewhere between Mitsuba and Brigade, and it uses the best of both worlds. Please note that using the GPU implies some limitations not expected:

- No `recursive` bsdfs. There are however some bsdf types like `coating` and `blend` which can use only SOME of the other bsdfs, specifically all those defined in BSDF_SIMPLE.h.
- No `recursive` textures. It is not possible to combine textures in mathematical functions to create complex appearances.
- No support for samplers. Only standard random number generators are used leading to unfavourable noise reduction.

All of these features are definitely possible in CUDA, although performance has to be monitored. There was no request for these complex materials up to now, so this was not actively worked on.

There are however some small features which are of less importance in Mitsuba:

- Adaptive Progressive Photon Mapping
- The main 3 volumetric estimators from *A Comprehensive Theory of Volumetric Radiance Estimation using Photon Points and Beams*
- Skeletal animation
- Completely dynamic deformable meshes and a dynamic scene with thousands of dynamic objects
- Parallax Occlusion Mapping and an (unrealistic but cool looking) orthogonal area light
- Dispersion (with Cauchy or Sellmeier model for index of refraction)

### Note

Personal note from the author: I started this project with very limited knowledge of C++ in general and particularly no knowledge of the stl. Because of this, there are definitely parts of the code which should not have been written this way. I am constantly trying to get rid of these legacy issues! The same applies to the git history; the first two years I was misusing the system (sparse commits with tons of changes).
Please see the wiki for further issues and comments on the code.


### License
This library is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License Version 3 as published by the Free Software Foundation.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.