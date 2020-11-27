#pragma once

#include <fstream>

namespace CudaFractals
{
    class Tests
    {
        private:

        public:
        static void testTriangleS(std::string, int);
        static void testTriangleP(std::string, int);
        static void testMandelbrotS(std::string, int);
        static void testMandelbrotP(std::string, int);

        static void runAllTests();
        static void runDepthTest();
        static void runResTest();
        static void runBlockTest();
    };
}
