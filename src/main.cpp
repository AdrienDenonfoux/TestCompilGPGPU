
// #include "cpp_test_compil.hpp"
// #include "cuda_test_compil.hpp"
// #include "TD1.hpp"
// #include "TP2.hpp"

#include <cmath>

#include <algorithm>
#include <random>
#include <vector>

#include <iostream>

std::vector<int> generateRandomNumbers(int limitMax, int nbNumbers)
{
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, limitMax);
  std::vector<int> randomNumbers(nbNumbers);
  std::generate(randomNumbers.begin(), randomNumbers.end(), [&] {
    return distribution(generator);
  });
  return randomNumbers;
}

int main(int, char*[])
{
  // runOnCPU();
  // runOnGPU();
  // nx2_plus_my();
  // nombre_premier();

  const int limitMax = 10000;
  // auto primeNumbers = generatePrimeNumbers(std::sqrt(limitMax));
  // std::cout << "Prime numbers: ";
  // for (auto prime : primeNumbers)
  // {
  //   std::cout << prime << ", ";
  // }
  // std::cout << std::endl;
 
  const int nbNumbers = 50000;
  auto randomNumbers = generateRandomNumbers(limitMax, nbNumbers);
  std::cout << "Randoms numbers: ";
  for (auto nb : randomNumbers)
  {
    std::cout << nb << ", ";
  }
  std::cout << std::endl;
  // auto randomPrimeNumbers = convolution(randomNumbers, primeNumbers);
  std::cout << "Randoms prime numbers: ";
  for (auto nb : randomNumbers)
  {
    std::cout << nb << ", ";
  }
  std::cout << std::endl;
 
  return 0;
}