import os
import subprocess
import argparse

class library:
    def __init__(self, name: str, url: str, commit: str) -> None:
        self.name = name
        self.url = url
        self.commit = commit

    def clone(self, dir: str) -> None:
        if os.path.exists(os.path.join(dir, self.name)):
            print( "========== " + self.name + " already exists ==========" )
            return

        print( "========== clone " + self.name + " ==========" )
        subprocess.call( ['git', 'clone', self.url, self.name], cwd=dir )
        subprocess.call( ['git', 'checkout', self.commit, '-q'], cwd=os.path.join(dir, self.name) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=True)
    args = parser.parse_args()

    libs = [
        library('stb', 'https://github.com/nothings/stb.git', 'af1a5bc352164740c1cc1354942b1c6b72eacb8a'),
        library('tinygltf', 'https://github.com/syoyo/tinygltf.git', 'aaf631c984c6e725573840c193ca2ff0ea216e7b'),
        library('tinyply', 'https://github.com/ddiakopoulos/tinyply.git', 'e5d969413b8612de31bf96604c95bf294d406230'),
        library('tinyobj', 'https://github.com/tinyobjloader/tinyobjloader.git', 'cab4ad7254cbf7eaaafdb73d272f99e92f166df8'),
        library('vulkan', 'https://github.com/KhronosGroup/Vulkan-Headers.git', '2b55157592bf4c639b76cc16d64acaef565cc4b5'),
        library('imgui', 'https://github.com/ocornut/imgui.git', '4b654db9040851228857528b44e195e358868e9a'),
        library('imGuIZMO.quat', 'https://github.com/BrutPitt/imGuIZMO.quat.git', '6c038a90fdadae580b357fbaf26f83cafeb83a6a'),
        library('glfw', 'https://github.com/glfw/glfw.git', '3fa2360720eeba1964df3c0ecf4b5df8648a8e52'),
    ]
    for lib in libs:
        lib.clone(args.directory)

    glfwBuildDir = os.path.join(args.directory, 'glfw', 'build')
    if not os.path.exists(glfwBuildDir):
        print("========== build glfw ==========")
        subprocess.run( ' '.join(['mkdir', glfwBuildDir]), shell=True)
        subprocess.run( ' '.join(['cmake', '-B', glfwBuildDir, '-S', os.path.join(args.directory, 'glfw'), '-DBUILD_SHARED_LIBS=ON']), shell=True)
        subprocess.run( ' '.join(['cmake', '--build', glfwBuildDir, '--config', 'Release']), shell=True)
        subprocess.run( ' '.join(['cmake', '--build', glfwBuildDir, '--config', 'Debug']), shell=True)
