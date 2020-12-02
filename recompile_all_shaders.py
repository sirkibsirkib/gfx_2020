import os, glob, subprocess, time, sys
script_path = os.path.dirname(os.path.realpath(__file__));
shader_files =      glob.glob(script_path + "/src/*/*.vert", recursive=False)
shader_files.extend(glob.glob(script_path + "/src/*/*.frag", recursive=False))
for shader_file in shader_files:
  print("compiling", shader_file)
  args = [
    "glslc",             # shaderc glsl->spir-v binary
    shader_file,         # input glsl path
    "-o",                # output path flag
    shader_file + ".spv",# output spir-v path
    "-O"                 # optimize for performance
  ];
  subprocess.run(args)
input("Blocking until newline...");
