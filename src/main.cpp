
// #include "cpp_test_compil.hpp"
// #include "cuda_test_compil.hpp"
// #include "TD1.hpp"
// #include "TP2.hpp"
#include "TP_note.hpp"

#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>

// Fonction pour afficher une matrice sous forme de grille
void printMatrix(const std::vector<char>& matrix, int width) {
  for (size_t i = 0; i < matrix.size(); i++) {
      if (i % width == 0) std::cout << "\n";
      std::cout << (int)matrix[i] << " ";
  }
  std::cout << "\n";
}

int main(int, char*[])
{
  // runOnCPU();
  // runOnGPU();
  // nx2_plus_my();
  // nombre_premier();
  

  const int width = 5;
  const std::vector<char> image = {
    10, 10, 10, 10, 10,
    10, 10, 10, 10, 10,
    10, 10, 50, 10, 10,
    10, 10, 10, 10, 10,
    10, 10, 10, 10, 10,
  };
  const int widthMask = 3;
  const std::vector<char> mask = {
    0, -1,  0,
   -1,  5, -1,
    0, -1,  0
  };

  std::cout << "Image :";
  printMatrix(image,width);

  // std::cout << "\nMasque :";
  // printMatrix(mask, 3);

  std::vector<char> imageTraite = convolution(image, width, mask, widthMask);

  std::cout << "Image traite :";
  printMatrix(imageTraite,width);
 
  return 0;
}