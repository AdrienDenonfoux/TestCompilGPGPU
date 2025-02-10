
#include "cuda_helper.hpp"

#include <cuda_runtime.h>
#include <vector>

namespace {

__global__ void filtreNbPremiers( 
                    int * inoutListeNombresAleatoires, const int nbNombresAleatoires,
              const int * listeNombresPremiers,      const int nbNombresPremiers )
{
  extern __shared__ int listeNbresPremiersShrd[];
  int indiceGlobal = threadIdx.x + blockIdx.x * blockDim.x;

  if ( threadIdx.x < nbNombresPremiers )
  {
    listeNbresPremiersShrd[ threadIdx.x ] = listeNombresPremiers[ threadIdx.x ];
  }

  __syncthreads();
  if ( indiceGlobal < nbNombresAleatoires )
  {
    int nombreAleatoire = inoutListeNombresAleatoires[ indiceGlobal ];
    bool estNombrePremier = true;

    int i = 0;
    while ( estNombrePremier && i < nbNombresPremiers )
    {
      estNombrePremier = !(nombreAleatoire != listeNbresPremiersShrd[ i ] && (nombreAleatoire % listeNbresPremiersShrd[ i ] == 0));
      ++i;
    }
    if ( !estNombrePremier )
    {
      inoutListeNombresAleatoires[ indiceGlobal ] = -1;
    }
  }
}

} // namespace

std::vector<char> convolution(
    const std::vector<char> & image, const int width,
    const std::vector<char> & mask, const int widthMask)

{
//   dim3 dimBloc( 512 );
//   dim3 dimGrille( randomNumbers.size() / dimBloc.x + ( randomNumbers.size() % dimBloc.x != 0 ) );
// 	int tailleShared = primeNumbers.size() * sizeof( int );

//   {
//     MemoryCopier randomNumberCopier( randomNumbers.data(), randomNumbers.size(), CopyType::BothSide );
//     MemoryCopier primeNumbersCopier( primeNumbers.data(), primeNumbers.size(), CopyType::CpuToGpuOnly);
//     filtreNbPremiers<<< dimGrille, dimBloc, tailleShared >>>(
//       randomNumberCopier.gpuData(),
//       randomNumbers.size(),
//       primeNumbersCopier.gpuData(),   
//       primeNumbers.size()
//     );
//     exitOnError( cudaGetLastError() );
//   }
//   randomNumbers.erase(std::remove(randomNumbers.begin(), randomNumbers.end(), -1), randomNumbers.end());
    std::vector<char> out;
  return out;
}
