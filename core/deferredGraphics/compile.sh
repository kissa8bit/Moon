full_path=$(realpath $0)
dir_path=$(dirname $full_path)

python3 $dir_path/../../scripts/shaders_compile.py -d $dir_path/shaders -o $dir_path/spv -c $dir_path/../../dependences/vulkan_tools/glslc
