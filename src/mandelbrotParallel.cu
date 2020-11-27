#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <fstream>
#include <chrono>

#include "GLManager.hpp"
#include "constants.hpp"
#include "cudaCheck.hpp"
#include "mandelbrotParallel.hpp"

using namespace CudaFractals::Parallel;

float3 *Mandelbrot::hostArr = nullptr;
float3 *Mandelbrot::devArr = nullptr;
int Mandelbrot::height;
int Mandelbrot::width;
double Mandelbrot::time;
int Mandelbrot::blockSize = 512;

__device__ inline point_t scalePoint(const point_t point, const int width, const int height) {
  return {xScaleMandelbrotStart + point.x * (xScaleMandelbrotWidth / (float)width),
          yScaleMandelbrotStart + point.y * (yScaleMandelbrotWidth / (float)height)};
}

__device__ inline int calcMandelbrotPoint(const point_t point, const int width, const int height,
                                          const int maxIter) {
  auto scaled = scalePoint(point, width, height);
  float x0 = scaled.x;
  float y0 = scaled.y;
  float x, y, x2, y2, w;
  int i = 0;

  x = y = x2 = y2 = w = 0.0f;

  while (x2 + y2 <= 4.0f && i++ < maxIter) {
    x = x2 - y2 + x0;
    y = w - x2 - y2 + y0;
    x2 = x * x;
    y2 = y * y;
    w = (x + y) * (x + y);
    i++;
  }

  return i;
}

__global__ void calcMandelbrot(float3 *dst, const int width, const int height, const int maxIter) {
  const float3 colors = {0.005f, 0.005f, 0.01f};
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int ix = pixel % width;
  const int iy = (pixel - ix) / width;

  if (ix < width && iy < height) {
    int m = calcMandelbrotPoint({(float)ix, (float)iy}, width, height, maxIter);
    m = m > maxIter ? 0 : maxIter - m;

    if (m) {
      dst[pixel].x = m * colors.x;
      dst[pixel].y = m * colors.y;
      dst[pixel].z = m * colors.z;
    } else {
      dst[pixel].x = 0;
      dst[pixel].y = 0;
      dst[pixel].z = 0;
    }
  }
}

void Mandelbrot::renderFunction(int limit) {
  if (limit <= 0) limit = 100;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point stop;
  start = std::chrono::high_resolution_clock::now();

  width = GLManager::getWidth();
  height = GLManager::getHeight();

  delete[] hostArr;
  hostArr = new float3[width * height];
  CUDA_CHECK(cudaMalloc((void **)&devArr, width * height * sizeof(float3)));

  dim3 blockDims(blockSize, 1, 1);
  dim3 gridDims((unsigned int)ceil((double)(width * height / blockDims.x)), 1, 1);

  calcMandelbrot<<<gridDims, blockDims>>>(devArr, width, height, limit);
  CUDA_CHECK(cudaMemcpy(hostArr, devArr, width * height * sizeof(float3),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(devArr));

  stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	time = duration.count();	
}

void Mandelbrot::draw(void) {
  int counter = 0;
  float3 *col;

  glClear(GL_COLOR_BUFFER_BIT);
  glPointSize(1.0f);
  glBegin(GL_POINTS);
  for (int y = -(height / 2); y < height / 2; y++) {
    for (int x = -(width / 2); x < width / 2; x++) {
      col = hostArr + counter;
      glColor3f(col->x, col->y, col->z);
      glVertex2i(x, y);
      counter++;
    }
  }

  glEnd();
  glFlush();
}

double Mandelbrot::getTime()
{
  return time;
}

void Mandelbrot::setBlockSize(int size)
{
  if(size > 0) blockSize = size;
}
