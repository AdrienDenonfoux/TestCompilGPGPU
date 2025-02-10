
#include "cuda_helper.hpp"

#include <iostream>

void exitOnError(const cudaError_t error)
{
  if (error != cudaSuccess)
  {
    std::cerr << "Error on cuda call: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
}