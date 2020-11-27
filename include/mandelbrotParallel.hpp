#pragma once

#include <vector_types.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif
#include <GL/gl.h>

namespace CudaFractals
{
    namespace Parallel
    {
        struct point_t
        {
            float x;
            float y;
        };

        class Mandelbrot
        {
        public:
            Mandelbrot() = default;
            ~Mandelbrot() = default;

            static void renderFunction(int limit);
            static void draw();
            static double getTime();

        private:
            static float3 *hostArr;
            static float3 *devArr;
            static int height;
            static int width;
            static double time;
        };

    } // namespace Parallel
} // namespace CudaFractals
