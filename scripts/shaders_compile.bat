set dir_path=%~dp0

python %dir_path%/shaders_compile.py -d %dir_path%/../core/deferredGraphics/shaders -o %dir_path%/../core/deferredGraphics/spv -c %dir_path%/../dependences/vulkan_tools/glslc.exe

python %dir_path%/shaders_compile.py -d %dir_path%/../core/workflows/shaders -o %dir_path%/../core/workflows/spv -c %dir_path%/../dependences/vulkan_tools/glslc.exe

python %dir_path%/shaders_compile.py -d %dir_path%/../core/rayTracingGraphics/shaders -o %dir_path%/../core/rayTracingGraphics/spv -c %dir_path%/../dependences/vulkan_tools/glslc.exe
