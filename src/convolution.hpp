#pragma once

#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>

std::vector<unsigned char> convolution_CPU(
    const std::vector<unsigned char> & image, const int width,
    const std::vector<char> & mask, const int widthMask);