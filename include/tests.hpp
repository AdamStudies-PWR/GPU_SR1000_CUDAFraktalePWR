#pragma once

#include <fstream>

namespace CudaFractals
{
    class Tests
    {
        private:
        static void testTriangleS(std::string filename, int depth);
        static void testTriangleP(std::string filename, int depth);
        static void testMandelbrotS(std::string filename, int depth);
        static void testMandelbrotP(std::string filename, int depth);

        public:
        static void singleTriangleS(int res, int depth);
        static void singleTriangleP(int res, int block, int depth);
        static void singleMandelbrotS(int res, int depth);
        static void singleMandelbrotP(int res, int block, int depth);

        static void runDepthTest();
        static void runResTest();
        static void runBlockTest();
        static void runAllTests();
    };
}
