cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(VirtualEndoscopy)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/VirtualEndoscopy")
set(EXTENSION_CATEGORY "MikeTools")
set(EXTENSION_CONTRIBUTORS "Mike Bindschadler (Seattle Children's Hospital), ")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/VirtualEndoscopy.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/VirtualEndoscopy/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(VirtualEndoscopy)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
