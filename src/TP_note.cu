#include "cuda_helper.hpp"

#include "convolution.cpp"

#include <cuda_runtime.h>

namespace {

    // Kernel CUDA pour appliquer la convolution sur GPU
    __global__ void convolution_GPU( 
        unsigned char* paddedImage, int width, int height,
        char* mask, int widthMask,
        unsigned char* output,
        int maskRadius, int paddedWidth)
    {
        // Utilisation de la mémoire partagé. Dans notre cas le masque est une variable qui ne change pas entre les threads.
        // C'est pour cela qu'on l'a place en mémoire partagée pour améliorer la l'accès en mémoire.
        extern __shared__ char maskShared[];
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
    
        // Charger le masque en mémoire partagée
        if (threadIdx.x < widthMask && threadIdx.y < widthMask)
        {
            int maskIndex = threadIdx.y * widthMask + threadIdx.x;
            maskShared[maskIndex] = mask[maskIndex];
        }
    
        __syncthreads(); // Synchronisation pour garantir que tous les threads ont chargé le masque
    
        // Vérification des bornes de l'image
        if (x < width && y < height)
        {
            int sum = 0;
            
            // Appliquer la convolution
            for (int j = -maskRadius; j <= maskRadius; ++j) {
                for (int i = -maskRadius; i <= maskRadius; ++i) {
                    int imageIndex = (y + j + maskRadius) * paddedWidth + (x + i + maskRadius);
                    int maskIndex = (j + maskRadius) * widthMask + (i + maskRadius);
    
                    sum += paddedImage[imageIndex] * maskShared[maskIndex];
                }
            }
    
            // Normalisation du résultat dans la plage [0, 255]
            output[y * width + x] = static_cast<char>(max(0, min(255, sum)));
        }
    }

} // namespace

// Implémentation d'une convolution d'une image avec un masque en utilisant CUDA
std::vector<unsigned char> convolution(
    const std::vector<unsigned char> &image, int width,
    const std::vector<char> &mask, int widthMask)
{
    int height = image.size() / width; // Calcul de la hauteur de l'image
    int maskRadius = widthMask / 2; // Rayon du masque
    int paddedWidth = width + 2 * maskRadius; // Largeur après padding
    int paddedHeight = height + 2 * maskRadius; // Hauteur après padding
    std::vector<char> mask_GPU = mask; // Copie du masque pour le GPU

    // Définition des tailles de grille et de blocs pour CUDA
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    int tailleShared = mask.size() * sizeof(char); // Taille de la mémoire partagée

    std::vector<unsigned char> output(image.size(), 0); // Vecteur de sortie
    std::vector<unsigned char> paddedImage(paddedWidth * paddedHeight, 0); // Vecteur de l'image paddée

    // Copier l'image originale dans l'image paddée
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            paddedImage[(y + maskRadius) * paddedWidth + (x + maskRadius)] = image[y * width + x];
        }
    }

    {
        // Gestion mémoire GPU avec MemoryCopier de vos fonctions cuda_helper.cpp
        MemoryCopier imageCopier(paddedImage.data(), paddedImage.size(), CopyType::BothSide);
        MemoryCopier maskCopier(mask_GPU.data(), mask.size(), CopyType::CpuToGpuOnly);
        MemoryCopier outputCopier(output.data(), output.size(), CopyType::BothSide);

        // Lancement du kernel CUDA
        convolution_GPU<<<gridSize, blockSize, tailleShared>>>(
            imageCopier.gpuData(),
            width, height,
            maskCopier.gpuData(),
            widthMask,
            outputCopier.gpuData(),
            maskRadius,
            paddedWidth
        );
        exitOnError(cudaGetLastError()); // Vérification des erreurs CUDA
    }

    return output; // Retourne l'image convoluée
}
