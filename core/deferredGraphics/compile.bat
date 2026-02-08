set dir_path=%~dp0

python %dir_path%/../../scripts/shaders_compile.py -d %dir_path%/shaders -o %dir_path%/spv -c %dir_path%/../../dependences/vulkan_tools/glslc.exe
