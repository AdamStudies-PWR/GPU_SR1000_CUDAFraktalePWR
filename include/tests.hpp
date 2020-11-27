#pragma once

#include <fstream>

namespace CudaFractals
{
    class Tests
    {
        private:

        public:
        static void testTriangleS(int);
        static void testTriangleP(int);
        static void testMandelbrotS(int);
        static void testMandelbrotP(int);

        static void runAllTests(int, int);
    };
}
