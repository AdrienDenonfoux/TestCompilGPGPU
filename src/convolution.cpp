#include "convolution.hpp"

std::vector<char> convolution_CPU(
    const std::vector<char> & image, const int width,
    const std::vector<char> & mask, const int widthMask)
{
    int height = image.size() / width;
    int maskRadius = widthMask / 2;
    int paddedWidth = width + 2 * maskRadius;
    int paddedHeight = height + 2 * maskRadius;

    std::vector<char> paddedImage(paddedWidth * paddedHeight, 0);

    // Copier l'image originale dans l'image paddée
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            paddedImage[(y + maskRadius) * paddedWidth + (x + maskRadius)] = image[y * width + x];
        }
    }

    for (size_t i = 0; i < paddedImage.size(); i++) {
        if (i % paddedWidth == 0) std::cout << "\n";
        std::cout << (int)paddedImage[i] << " ";
    }
    std::cout << "\n";

    std::vector<char> output(image.size(), 0);

    // Appliquer la somme pondérée des pixels suivant le masque
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;

            for (int j = -maskRadius; j <= maskRadius; ++j) {
                for (int i = -maskRadius; i <= maskRadius; ++i) {
                    int imageIndex = (y + j + maskRadius) * paddedWidth + (x + i + maskRadius);
                    int maskIndex = (j + maskRadius) * widthMask + (i + maskRadius);
                
                    sum += paddedImage[imageIndex] * mask[maskIndex];

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