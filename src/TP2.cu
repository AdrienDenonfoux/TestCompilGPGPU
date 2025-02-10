
#include "TP2.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <random>

namespace {

__global__ void nombre_premier_GPU(int * nbPremier, int * tabNombre, int * tabOut, int tailleTab)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idGlobal = idx + blockDim.x * gridDim.x * idy;
  
  
  if(idGlobal < tailleTab){
    tabOut[idGlobal] = -1;
    for (int i = 0; i <= tailleTab; ++i) {
      if(nbPremier[i] == tabNombre[idGlobal])
        tabOut[idGlobal] = tabNombre[idGlobal];
    }
  }
}

} // namespace

void cribleEratosthene(int tableau[], int limite) {
    // Initialiser un tableau booléen pour marquer les nombres premiers
    bool* estPremier = new bool[ limite + 1 ];
    for (int i = 0; i <= limite; ++i) {
        estPremier[i] = true;  // Supposer que tous les nombres sont premiers
    }
    estPremier[0] = estPremier[1] = false;  // 0 et 1 ne sont pas premiers

    // Application du crible d'Ératosthène
    for (int i = 2; i * i <= limite; ++i) {
        if (estPremier[i]) {
            for (int j = i * i; j <= limite; j += i) {
                estPremier[j] = false;  // Marquer les multiples de i comme non premiers
            }
        }
    }

    // Remplir le tableau des résultats avec les nombres premiers
    int index = 0;
    for (int i = 2; i <= limite; ++i) {
        if (estPremier[i]) {
            tableau[index++] = i;  // Ajouter les nombres premiers dans le tableau
        }
    }
}

void nombre_premier()
{
  dim3 dimGrille( 1000, 1000 );
  dim3 dimBloc( 512, 2 );

  int limite;
  std::cout << "Entrez la limite superieure pour generer les nombres premiers : ";
  std::cin >> limite;

  int* nbPremier = new int[ limite + 1 ];

  cribleEratosthene(nbPremier, limite);

  std::cout << "Les nombres premiers jusqu'à " << limite << " sont : ";
    
  // Affichage des nombres premiers stockés dans le tableau
  for (int i = 0; i < limite; ++i) {
      if (nbPremier[i] != 0 && nbPremier[i] != -842150451) {  // On affiche seulement les éléments non nuls (les premiers)
          std::cout << nbPremier[i] << " ";
      }
  }
  std::cout << std::endl;
 

  int tailleTab;
  std::cout << "Entrez la taille du tableau de nombre aleatoire : ";
  std::cin >> tailleTab;

  int* tabNombre = new int[ tailleTab ];

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1,limite);
  for (int i = 0; i < tailleTab; ++i){
    tabNombre[i] = distribution(generator);
  }

  std::cout << "Les " << tailleTab << " nombres aleatoires sont : ";
  for (int i = 0; i < tailleTab; ++i) {
        std::cout << tabNombre[i] << " ";
  }
  std::cout << std::endl;

  // Tab GPU
  int* nbPremier_GPU;
  int* tabNombre_GPU;
  int* tabOut_GPU;
  int* tabOut = new int[ tailleTab ];


  cudaMalloc(&nbPremier_GPU, tailleTab * sizeof(int));
  cudaMalloc(&tabNombre_GPU, tailleTab * sizeof(int));
  cudaMalloc(&tabOut_GPU, tailleTab * sizeof(int));

  // Copie vers GPU
  cudaMemcpy(nbPremier_GPU, nbPremier, tailleTab * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(tabNombre_GPU, tabNombre, tailleTab * sizeof(int), cudaMemcpyHostToDevice);


  // Appel kernel
  nombre_premier_GPU<<< dimGrille, dimBloc >>>(nbPremier_GPU, tabNombre_GPU, tabOut_GPU, tailleTab);

  // Copie vers CPU
  cudaMemcpy( tabOut, tabOut_GPU, tailleTab * sizeof( int ), cudaMemcpyDeviceToHost );

  printf("\nTab Out : ");
  for (int i = 0; i < tailleTab; ++i){
    if (tabOut[i] != -1){
      printf(" %d",tabOut[i]);
    }
  }

  cudaFree(nbPremier_GPU);
  cudaFree(tabNombre_GPU);
  cudaFree(tabOut_GPU);

  delete(tabNombre);
  delete(nbPremier);
  delete(tabOut);
  auto err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error on nx2_plus_my: " << cudaGetErrorString(err) << std::endl;
  }
}
