#include "convolution.hpp"

/*
    Ce code effectue une convolution sur une image en niveaux de gris en utilisant un masque (filtre).
    Il fonctionne en mode CPU et applique un remplissage (padding) pour éviter les effets de bord.
*/
std::vector<unsigned char> convolution_CPU(
    const std::vector<unsigned char> & image, const int width,
    const std::vector<char> & mask, const int widthMask)
    {
        int height = image.size() / width; // Calcul de la hauteur de l'image
        int maskRadius = widthMask / 2; // Rayon du masque pour le padding
        int paddedWidth = width + 2 * maskRadius; // Largeur de l'image après padding
        int paddedHeight = height + 2 * maskRadius; // Hauteur de l'image après padding
    
        std::vector<unsigned char> paddedImage(paddedWidth * paddedHeight, 0); // Création de l'image paddée remplie de 0
    
        // Copier l'image originale dans l'image paddée avec un décalage
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                paddedImage[(y + maskRadius) * paddedWidth + (x + maskRadius)] = image[y * width + x];
            }
        }
    
        // Affichage de l'image paddée pour le debug
        for (size_t i = 0; i < paddedImage.size(); i++) {
            if (i % paddedWidth == 0) std::cout << "\n";
            std::cout << (int)paddedImage[i] << " ";
        }
        std::cout << "\n";
    
        std::vector<unsigned char> output(image.size(), 0); // Création du vecteur de sortie initialisé à 0
    
        // Appliquer la convolution
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int sum = 0; // Somme accumulée pour la convolution
    
                // Parcourir le masque
                for (int j = -maskRadius; j <= maskRadius; ++j) {
                    for (int i = -maskRadius; i <= maskRadius; ++i) {
                        int imageIndex = (y + j + maskRadius) * paddedWidth + (x + i + maskRadius); // Index dans l'image paddée
                        int maskIndex = (j + maskRadius) * widthMask + (i + maskRadius); // Index dans le masque
                    
                        sum += paddedImage[imageIndex] * mask[maskIndex]; // Multiplication et addition des valeurs
    
                        std::cout << (int)sum << " "; // Debug de la somme courante
                        std::cout << "\n";
                    }
                }
                // Normalisation de la valeur pour rester dans [0, 255]
                output[y * width + x] = static_cast<char>(std::max(0, std::min(255, sum)));
            }
        }
        return output; // Retourne l'image convoluée
    }
    