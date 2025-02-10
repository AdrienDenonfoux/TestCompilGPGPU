#pragma once

#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>

std::vector<char> convolution_CPU(
    const std::vector<char> & image, const int width,
    const std::vector<char> & mask, const int widthMask);