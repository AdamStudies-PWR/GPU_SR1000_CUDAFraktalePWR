#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <fstream>

#include "GLManager.hpp"
#include "constants.hpp"
#include "cudaCheck.hpp"
#include "mandelbrotParallel.hpp"

using namespace CudaFractals::Parallel;

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

__global__ void calcMandelbrot(float3 *dst, const int width, const int height) {
  const float3 colors = {0.01f, 0.01f, 0.01f};
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int ix = pixel % width;
  const int iy = (pixel - ix) / width;
  const int maxIter = 100;

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

void Mandelbrot::renderFunction(void) {
  const int width = GLManager::getWidth();
  const int height = GLManager::getHeight();

  hostArr = new float3[width * height];
  CUDA_CHECK(cudaMalloc((void **)&devArr, width * height * sizeof(float3)));

  dim3 blockDims(512, 1, 1);
  dim3 gridDims((unsigned int)ceil((double)(width * height / blockDims.x)), 1, 1);

  calcMandelbrot<<<gridDims, blockDims>>>(devArr, width, height);
  CUDA_CHECK(cudaMemcpy(hostArr, devArr, width * height * sizeof(float3),
                        cudaMemcpyDeviceToHost));

  draw(width, height);

  CUDA_CHECK(cudaFree(devArr));
  delete[] hostArr;
}

void Mandelbrot::draw(const int width, const int height) {
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
  printf("Drawn\n");
}
