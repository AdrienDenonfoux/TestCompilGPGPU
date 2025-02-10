
#pragma once

#include <cuda_runtime.h>

void exitOnError(const cudaError_t error);

enum class CopyType
{
  BothSide,
  CpuToGpuOnly
};

template <typename T>
class MemoryCopier
{
  public:

    MemoryCopier(T* cpuData, int size, const CopyType type);
    ~MemoryCopier();

    T* gpuData() const;

  private:
    T* mCPUData = nullptr;
    int mSize = 0;
    CopyType mType;
    T* mGPUData = nullptr;
};

template <typename T>
MemoryCopier<T>::MemoryCopier(T* cpuData, int size, const CopyType type)
: mCPUData(cpuData), mSize(size), mType(type)
{
  if (mType == CopyType::BothSide || mType == CopyType::CpuToGpuOnly)
  {
    exitOnError( cudaMalloc( &mGPUData, mSize * sizeof( T ) ) );
    exitOnError( cudaMemcpy( mGPUData, mCPUData, mSize * sizeof( T ), cudaMemcpyHostToDevice ) );
  }
}

template <typename T>
MemoryCopier<T>::~MemoryCopier()
{
  if (mType == CopyType::BothSide)
  {
    exitOnError( cudaMemcpy( mCPUData, mGPUData, mSize * sizeof( T ), cudaMemcpyDeviceToHost ) );
  }
  exitOnError( cudaFree( mGPUData ) );
}

template <typename T>
T* MemoryCopier<T>::gpuData() const
{
  return mGPUData;
}