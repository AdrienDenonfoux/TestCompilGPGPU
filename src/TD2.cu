
#include "TD2.hpp"

#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void rgb_to_grey_GPU(char*** tabImage, char width, char height, char** tabOut)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < width && idy < height) {
    tabOut[idy][idx] = 0.299 * tabImage[idy][idx][0] + 0.587 * tabImage[idy][idx][1] + 0.114 * tabImage[idy][idx][2];
  }
}

} // namespace

void rgb_to_grey()
{
  dim3 dimGrille( 1000, 1000 );
  dim3 dimBloc( 32 , 32 );

  char width, height;
  std::cin >> width;
  std::cin >> height;

  // char*** ImageGPU;
  // char** ImageOutGPU;
  // Allocation
  // cudaMalloc(&tabX_GPU, tailleTab * sizeof(int));
  // cudaMalloc(&tabY_GPU, tailleTab * sizeof(int));
  // cudaMalloc(&tabOut_GPU, tailleTab * sizeof(int));

  // Copie vers GPU
  // cudaMemcpy(tabX_GPU, inX, tailleTab * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(tabX_GPU, inY, tailleTab * sizeof(int), cudaMemcpyHostToDevice);

  char*** tabImage = new char**[height];
  char** tabImageOut = new char*[height];
  for (int h = 0; h < height; ++h){
    tabImage[h] = new char*[width];
    tabImageOut[h] = new char[width];
    for (int w = 0; w < width; ++w)
      tabImage[width][height] = new char[3];
  }

  for (int x = 0; x < width; ++x){
    for (int y = 0; y < height; ++y){
      for (int rgb = 0; rgb < 2; rgb++){
        tabImage[x][y][rgb] = (x * 56 + y * 40 + rgb)%255;
      }
    }
  }
  

  rgb_to_grey_GPU<<< dimGrille, dimBloc >>>(tabImage, width, height, tabImageOut);
  auto err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error on nx2_plus_my: " << cudaGetErrorString(err) << std::endl;
  }
}
