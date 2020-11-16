#pragma once

#include <vector_types.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <GL/gl.h>

namespace CudaFractals {
namespace Parallel {

    struct point_t {
        float x;
        float y;
    };

    class Mandelbrot {
    public:
        Mandelbrot() = default;
        ~Mandelbrot() = default;

        void renderFunction();

    private:
        void draw(const int width, const int height);

        float3* hostArr = nullptr;
        float3* devArr = nullptr;
    };

}
}
