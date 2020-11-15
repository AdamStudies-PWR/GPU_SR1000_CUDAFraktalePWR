#pragma once

#include<complex>

#define LENGTH 100

namespace CudaFractals 
{
class MandelbrotSequential
{

private:
    int *fractal;
    int height;
    int width;
    int mandelbrot(std::complex<float> C);
    void draw();
public:
    void generateFractal();

};
}  // namespace CudaFractals