
#include "convolution.cpp"

#include <cuda_runtime.h>

namespace {

__global__ void convolution_GPU( 
            std::vector<char> image, const int width,
            std::vector<char> mask, const int widthMask,
            std::vector<char> output)
{
  extern __shared__ int maskShared[];
  int indiceGlobal = threadIdx.x + blockIdx.x * blockDim.x;

  if ( threadIdx.x < widthMask )
  {
    maskShared[ threadIdx.x ] = mask[ threadIdx.x ];
  }

  __syncthreads();
  if ( indiceGlobal < width )
  
    int maskRadius = widthMask / 2;
    int paddedWidth = width + 2 * maskRadius;

    // Appliquer la somme pondérée des pixels suivant le masque
    for (int y = 0; y < width; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;

            for (int j = -maskRadius; j <= maskRadius; ++j) {
                for (int i = -maskRadius; i <= maskRadius; ++i) {
                    int imageIndex = (y + j + maskRadius) * paddedWidth + (x + i + maskRadius);
                    int maskIndex = (j + maskRadius) * widthMask + (i + maskRadius);
                
                    sum += paddedImage[imageIndex] * maskShared[maskIndex];

                    std::cout << (int)sum << " ";
                    std::cout << "\n";
                }
            }
            // Attention ne pas dépasser 255 qui ai le niveau de gris le plus élevé.
            output[y * width + x] = static_cast<char>(std::max(0, std::min(255, sum)));
        }
    }
    return output;
}

} // namespace



// Implémentation d'une convolution d'une image avec un masque.
std::vector<char> convolution(
    const std::vector<char> & image, const int width,
    const std::vector<char> & mask, const int widthMask)
{
    dim3 dimBloc( 512 );
    dim3 dimGrille( image.size() / dimBloc.x + ( image.size() % dimBloc.x != 0 ) );
	int tailleShared = mask.size() * sizeof( int );

    std::vector<char> output;

    // Play on CPU
    //output = convolution_CPU(image, width, mask, widthMask);


    int maskRadius = widthMask / 2;
    int paddedWidth = width + 2 * maskRadius;

    std::vector<char> paddedImage(paddedWidth * paddedWidth, 0);

    // Copier l'image originale dans l'image paddée
    for (int y = 0; y < width; ++y) {
        for (int x = 0; x < width; ++x) {
            paddedImage[(y + maskRadius) * paddedWidth + (x + maskRadius)] = image[y * width + x];
        }
    }

    std::vector<char> output(image.size(), 0);

    std::vector<char> paddedImage_GPU;
    std::vector<char> mask_GPU;
    std::vector<char> output_GPU;


    cudaMalloc(&paddedImage_GPU, paddedWidth * sizeof(char));
    cudaMalloc(&mask_GPU, widthMask * sizeof(char));
    cudaMalloc(&output_GPU, width * sizeof(char));

    // Copie vers GPU
    cudaMemcpy(paddedImage, paddedImage, paddedWidth * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(mask, mask, widthMask * sizeof(char), cudaMemcpyHostToDevice);


    // Appel kernel
    convolution_GPU<<< dimGrille, dimBloc >>>(paddedImage_GPU, paddedWidth, mask_GPU, widthMask, output_GPU);

    // Copie vers CPU
    cudaMemcpy(output_GPU, output, width * sizeof( char ), cudaMemcpyDeviceToHost );

    return output;
}
