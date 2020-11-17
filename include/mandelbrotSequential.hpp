#pragma once

#include<complex>
#include <ctime>
#include <ratio>
#include <chrono>

namespace CudaFractals 
{
class MandelbrotSequential
{
private:
    static int LENGTH;
    static int height;
    static int width;
private:
    static int mandelbrot(std::complex<float> C);
public:
    static void draw();
    static double generateFractal(int length);
    static void setLength();

};
}  // namespace CudaFractals
