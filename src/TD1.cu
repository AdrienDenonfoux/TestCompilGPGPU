
#include "TD1.hpp"

#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void nx2_plus_my_GPU(int n, int m, int * tabX, int * tabY, int * tabOut, int tailleTab)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idGlobal = idx + blockDim.x * gridDim.x * idy;
  
  if(idGlobal < tailleTab){
    tabOut[idGlobal] = n * tabX[ idGlobal ] * tabX[ idGlobal ] + m * tabY[ idGlobal ];
  }

}

} // namespace

void nx2_plus_my()
{
  dim3 dimGrille( 1000, 1000 );
  dim3 dimBloc( 512, 2 );
  int tailleTab;
  int n;
  int m;

  std::cin >> tailleTab;
  std::cin >> n;
  std::cin >> m;

  // Tab GPU
  int* tabX_GPU;
  int* tabY_GPU;
  int* tabOut_GPU;

  // Tab CPU
  int* inX = new int[ tailleTab ];
  int* inY = new int[ tailleTab ];
  int* tabOut = new int[ tailleTab ];

  for (int i = 0; i < tailleTab; ++i){
    inX[i] = i + 1;
    inY[i] = tailleTab - i;
  }
  printf("Tab X : ");
  for (int i = 0; i < tailleTab; ++i){
    printf(" %d",inX[i]);
  }
  printf("\nTab Y : ");
  for (int i = 0; i < tailleTab; ++i){
    printf(" %d",inY[i]);
  }



  // Allocation
  cudaMalloc(&tabX_GPU, tailleTab * sizeof(int));
  cudaMalloc(&tabY_GPU, tailleTab * sizeof(int));
  cudaMalloc(&tabOut_GPU, tailleTab * sizeof(int));

  // Copie vers GPU
  cudaMemcpy(tabX_GPU, inX, tailleTab * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(tabX_GPU, inY, tailleTab * sizeof(int), cudaMemcpyHostToDevice);

  // Appel kernel
  nx2_plus_my_GPU<<< dimGrille, dimBloc >>>(n, m, tabX_GPU, tabY_GPU, tabOut_GPU, tailleTab);

  // Copie vers CPU
  cudaMemcpy( tabOut, tabOut_GPU, tailleTab * sizeof( int ), cudaMemcpyDeviceToHost );

  printf("\nTab Out : ");
  for (int i = 0; i < tailleTab; ++i){
    printf(" %d",tabOut[i]);
  }

  cudaFree(tabX_GPU);
  cudaFree(tabY_GPU);
  cudaFree(tabOut_GPU);
  auto err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error on nx2_plus_my: " << cudaGetErrorString(err) << std::endl;
  }
}
