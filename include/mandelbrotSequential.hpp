#pragma once

#include <vector>
#include <complex>
#include <ctime>
#include <ratio>
#include <chrono>

namespace CudaFractals 
{
class MandelbrotSequential
{
public:

    struct Colors
    {
        float r;
        float g;
        float b;
    };

private:
    static int LENGTH;
    static int height;
    static int width;
    static std::vector<Colors> colors;

    static int mandelbrotPoint(std::complex<float> C);

public:
    static void draw();
    static double generateFractal(int length);
};
}  // namespace CudaFractals
