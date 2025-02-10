
#pragma once

#include <vector>

std::vector<char> convolution(
    const std::vector<char> & image, const int width,
    const std::vector<char> & mask, const int widthMask);
