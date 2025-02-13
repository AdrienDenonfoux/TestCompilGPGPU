
# TP Filtrer une image

Les fichiers du TP sont :
- convolution.cpp/hpp : Convolution d'une image avec un masque exécutée sur CPU.
- cuda_helper.cpp/hpp : Les même que dans votre correction. Permet la gestion de l'allocation et copy pour fonction GPU.
- TP_note.cu/hpp : Une fonction convolution qui permet d'initialiser les variables et lancer la fonction convolution_GPU exécutée sur GPU.
- main.cpp : Convolution d'une image avec un masque.

# Test CUDA Compilation

## On Windows
- Install the CUDA toolkit
- Clone this repository
- Open as an administrator the x64 Native Tools Command
- Reach the cloned folder
- Build and run:
```console
mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug -G "Visual Studio 17 2022" -A x64 ..
cmake --build .
.\bin\Debug\Debug\TestCompilation.exe
```

Note:
- To use a newer version of CMake that the one provide with VS: `set PATH=C:\Program Files\CMake\bin;%PATH%`
- `-A x64` is used to specify the 64 bits arch that is the only one compatible with CUDA