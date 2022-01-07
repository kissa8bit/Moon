import os
import subprocess
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-c', '--compiler', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for entry in os.listdir(args.directory):
        if os.path.isdir(os.path.join(args.directory,entry)) and "__" not in entry:
            
            os.makedirs(os.path.join(args.output, entry), exist_ok=True)

            for file in os.listdir(os.path.join(args.directory, entry)):
                filename, file_extension = os.path.splitext(file)
                match file_extension:
                    case '.frag':
                        subprocess.run([args.compiler, os.path.join(args.directory, entry, file), '-o', os.path.join(args.output, entry, filename + 'Frag.spv')])
                    case '.vert':
                        subprocess.run([args.compiler, os.path.join(args.directory, entry, file), '-o', os.path.join(args.output, entry, filename + 'Vert.spv')])
                    case _:
                        pass