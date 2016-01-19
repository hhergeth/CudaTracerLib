<p align="center">
<a href='https://github.com/hhergeth/CudaTracerLib/wiki/Example-Renderings' style='font-size:2.2em'>Examples</span></a> <br>
<a href="https://github.com/hhergeth/CudaTracerLib/wiki/Example-Renderings">
<img src="http://hhergeth.markab.uberspace.de/Git-Wiki-Files/thumbnails/loadBox_20150716_080109_PPPM.jpg" style="width:880px;">
</a>
</p>

# CudaTracerLib

### Introduction
This library, for simplicity called CudaTracerLib, is a CUDA based ray tracer library implementing standard rendering algorithms like Path Tracing, Progressive Photon Mapping, Bidirectional Path Tracing and many more. It can be used in an offline render mode where only one frame is computed. The other option is to compute the frames directly into an Open GL/D3D buffer enabling "real-time" ray traced images.


### Building from Source
##### There are four dependencies:

- CUDA 7.5
- Boost, used for filesystem(compiled library!) and string functions
- FreeImage, used for loading and saving images.
- qMatrixLib, a tiny header only matrix library, available here https://github.com/hhergeth/qMatrixLib

##### For Windows users there is a Visual Studio 2013 project file available:

- Install the CUDA 7.5 toolkit from the official Nvidia site.
- Grab a copy of the precompiled boost libraries and extract them somewhere or compile them on your own.
- Download the FreeImage source code from [here](http://freeimage.sourceforge.net/download.html "FreeImage source code"). Use the provided Visual Studio solution file to build the static lib, but pay attention to set the compiler version to the same used for compiling this library!
- Download [qMatrixLib](https://github.com/hhergeth/qMatrixLib) and extract it somewhere handy.
- Set the include path of FreeImage, qMatrixLib and Boost under VC++ Directories/Include Directories
- Later when linking against CudaTracerLib add appropriate paths under VC++ Directories/Library Directories. Also go to Linker/Input/Additional Dependencies and add FreeImage.lib. Linking against the compiled Boost libraries requires no further action.

##### All other platforms can use the accompanying CMake file:

- Install the CUDA 7.5 toolkit from the official Nvidia site.
- Do the same for boost.
- There are multiple unofficial CMake versions of FreeImage available. Use one of these to compile the library or check if there are precompiled versions available as for example for Debian.
- Download [qMatrixLib](https://github.com/hhergeth/qMatrixLib) and extract it somewhere handy.
- Specify the paths of FreeImage, Boost (if necessary) and qMatrixLib (QMATRIX\_INCLUDE\_DIR) in CMake.

The microfacet and spectral probability distribution files from Mitsuba/PBRT are also necessary. They can be obtained from the [Mitsuba build](http://www.mitsuba-renderer.org/download.html) or [here from the repository](https://www.mitsuba-renderer.org/repos/mitsuba). Only the _data/ior_ and _data/microfacet_ folders are required.

Examples of how to use this library and an implementation of a custom rendering algorithm can be found in the [Github wiki](https://github.com/hhergeth/CudaTracerLib/wiki/Code-Examples).

### Acknowledgements
I would like to thank Wenzel Jakob for allowing me to use a lot of his work from Mitsuba - http://www.mitsuba-renderer.org. This includes the general interfaces and implementation of the `BSDF`, `Emitter`, `Sensor` classes. Furthermore I have used his `MicrofacetDistribution` and `RoughTransmittance` classes as well as the design of the `SamplingRecord` classes.

Thanks to Timo Aila and Samuli Laine for their research on BVH ray traversal on CUDA GPUs in *Understanding the Efficiency of Ray Traversal on GPUs*. I have used slight modifications of their code for the BVH computation as well as the traversal.


### License
This library is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License Version 3 as published by the Free Software Foundation.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.