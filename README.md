
<p align="center">
<a href='https://github.com/hhergeth/CudaTracerLib/wiki/Example-Renderings' style='font-size:2.2em'>Examples</span></a> <br>
<a href="https://github.com/hhergeth/CudaTracerLib/wiki/Example-Renderings">
<img src="http://hhergeth.markab.uberspace.de/Git-Wiki-Files/thumbnails/loadSanMiguel_20170812_023949_WavefrontPath.jpg" style="width:880px;">
</a>
</p>

# CudaTracerLib

### Introduction
This library, for simplicity called CudaTracerLib, is a CUDA based ray tracer library implementing standard rendering algorithms like Path Tracing, Progressive Photon Mapping, Bidirectional Path Tracing and many more.
For ease of usability the library can be compiled as an executable with a minimalistic main loop which uses the Mitsuba file loader to render most of the scenes found [here](https://benedikt-bitterli.me/resources/).


### Building from Source

- Install the CUDA 9.0 toolkit from the official Nvidia site.
- Clone the repository to your computer and use CMake to create build files (only x64 and v140 toolset for Visual Studio [see below]) for your system, all dependencies are handled automatically.
- The microfacet and spectral probability distribution files from Mitsuba/PBRT are also necessary. They can be obtained from the [Mitsuba build](http://www.mitsuba-renderer.org/download.html) or [here from the repository](https://github.com/mitsuba-renderer/mitsuba). Only the _data/ior_ and _data/microfacet_ folders are required.

Examples of how to use this library and an implementation of a custom rendering algorithm can be found in the [Github wiki](https://github.com/hhergeth/CudaTracerLib/wiki/Code-Examples).

### Remarks

- Under Windows one should use VS 2017 but CUDA currently requires the VS 2015 toolset, please make sure this is installed. It can be specified in CMake like shown [here](https://stackoverflow.com/questions/47154454/)
- Due to the Visual C++ compiler not being fully c++17 compatible the library currently employs workarounds to use the _filesystem_ and _optional_ libraries. This will be removed in the near future.

### Acknowledgements
I would like to thank Wenzel Jakob for allowing me to use a lot of his work from Mitsuba - http://www.mitsuba-renderer.org. This includes the general interfaces and implementation of the `BSDF`, `Emitter`, `Sensor` classes. Furthermore I have used his `MicrofacetDistribution` and `RoughTransmittance` classes as well as the design of the `SamplingRecord` classes.

Thanks to Timo Aila and Samuli Laine for their research on BVH ray traversal on CUDA GPUs in *Understanding the Efficiency of Ray Traversal on GPUs*. I have used slight modifications of their code for the BVH computation as well as the traversal.


### License
This library is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License Version 3 as published by the Free Software Foundation.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
