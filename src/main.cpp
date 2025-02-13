#include "TP_note.hpp"

#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>

// Fonction pour afficher une matrice sous forme de grille
void printMatrix(const std::vector<unsigned char>& matrix, int width) {
  for (size_t i = 0; i < matrix.size(); i++) {
      if (i % width == 0) std::cout << "\n";
      std::cout << (int)matrix[i] << " ";
  }
  std::cout << "\n";
}

// Surcharge de la fonction printMatrix pour les matrices de type char
void printMatrix(const std::vector<char>& matrix, int width) {
  for (size_t i = 0; i < matrix.size(); i++) {
      if (i % width == 0) std::cout << "\n";
      std::cout << (int)matrix[i] << " ";
  }
  std::cout << "\n";
}

int main(int, char*[])
{
  // Déclaration de la largeur de l'image
  const int width = 5;
  
  // Définition d'une image de test sous forme de tableau 5x5
  const std::vector<unsigned char> image = {
    100, 100, 100, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 150, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 100, 100, 100,
  };
  
  // Déclaration de la largeur du masque de convolution
  const int widthMask = 3;
  
  // Définition d'un masque de convolution (filtre de netteté)
  const std::vector<char> mask = {
    0, -1,  0,
   -1,  5, -1,
    0, -1,  0
  };

  // Affichage de l'image originale
  std::cout << "Image :";
  printMatrix(image, width);

  // Affichage du masque de convolution
  std::cout << "\nMasque :";
  printMatrix(mask, widthMask);

  // Application de la convolution sur l'image
  std::vector<unsigned char> imageTraite = convolution(image, width, mask, widthMask);

  // Affichage de l'image traitée après convolution
  std::cout << "Image traite :";
  printMatrix(imageTraite, width);
 
  return 0;
}
