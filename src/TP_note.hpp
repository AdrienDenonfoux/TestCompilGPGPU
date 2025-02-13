
#pragma once

#include <vector>

std::vector<unsigned char> convolution(
    const std::vector<unsigned char> & image, const int width,
    const std::vector<char> & mask, const int widthMask);
