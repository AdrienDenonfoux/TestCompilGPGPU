
#include "cuda_helper.hpp"

#include "convolution.cpp"

#include <cuda_runtime.h>

namespace {

    __global__ void convolution_GPU( 
        char* paddedImage, int width, int height,
        char* mask, int widthMask,
        char* output,
        int maskRadius, int paddedWidth)
    {
        extern __shared__ char maskShared[];
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
    
        if (threadIdx.x < widthMask && threadIdx.y < widthMask)
        {
            int maskIndex = threadIdx.y * widthMask + threadIdx.x;
            maskShared[maskIndex] = mask[maskIndex];
        }
    
        __syncthreads();
    
        if (x < width && y < height)
        {
            int sum = 0;
            
            for (int j = -maskRadius; j <= maskRadius; ++j) {
                for (int i = -maskRadius; i <= maskRadius; ++i) {
                    int imageIndex = (y + j + maskRadius) * paddedWidth + (x + i + maskRadius);
                    int maskIndex = (j + maskRadius) * widthMask + (i + maskRadius);
    
                    sum += paddedImage[imageIndex] * maskShared[maskIndex];
                }
            }
    
            output[y * width + x] = static_cast<char>(max(0, min(255, sum)));
        }
    }

} // namespace



// Implémentation d'une convolution d'une image avec un masque.
std::vector<char> convolution(
    const std::vector<char> &image, int width,
    const std::vector<char> &mask, int widthMask)
{
    int height = image.size() / width;
    int maskRadius = widthMask / 2;
    int paddedWidth = width + 2 * maskRadius;
    int paddedHeight = height + 2 * maskRadius;
    std::vector<char> mask_GPU = mask;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    int tailleShared = mask.size() * sizeof(char);

    std::vector<char> output(image.size(), 0);
    std::vector<char> paddedImage(paddedWidth * paddedHeight, 0);

    // Copier l'image originale dans l'image paddée
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            paddedImage[(y + maskRadius) * paddedWidth + (x + maskRadius)] = image[y * width + x];
        }
    }

    {
        MemoryCopier imageCopier(paddedImage.data(), paddedImage.size(), CopyType::BothSide);
        MemoryCopier maskCopier(mask_GPU.data(), mask.size(), CopyType::CpuToGpuOnly);
        MemoryCopier outputCopier(output.data(), output.size(), CopyType::BothSide);

        convolution_GPU<<<gridSize, blockSize, tailleShared>>>(
            imageCopier.gpuData(),
            width, height,
            maskCopier.gpuData(),
            widthMask,
            outputCopier.gpuData(),
            maskRadius,
            paddedWidth
        );
        exitOnError(cudaGetLastError());
    }

    return output;
}