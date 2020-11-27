#include "tests.hpp"
#include "STriangleSeq.hpp"
#include "STrianglePar.hpp"
#include "mandelbrotSequential.hpp"
#include "mandelbrotParallel.hpp"

using namespace CudaFractals;

void Tests::testTriangleS(std::string filename, int depth)
{
    filename.append("trianglesequential.txt");
    std::ofstream file(filename);

    for(int i=0; i<100; i++)
    {
        STriangleSeq::Generate(depth);
        file << STriangleSeq::GetTime()*1000 << " ms" << std::endl;
    }

    file.close();
}

void Tests::testTriangleP(std::string filename, int depth)
{
    filename.append("triangleparallel.txt");
    std::ofstream file(filename);

    for(int i=0; i<100; i++)
    {
        STrianglePar::Generate(depth);
        file << STrianglePar::GetTime()*1000 << " ms" << std::endl;
    }

    file.close();
}

void Tests::testMandelbrotS(std::string filename, int depth)
{
    filename.append("mandelbrotsequential.txt");
    std::ofstream file(filename);

    for(int i=0; i<100; i++)
    {
        file << MandelbrotSequential::generateFractal(depth)*1000 << " ms" << std::endl;
    }

    file.close();
}

void Tests::testMandelbrotP(std::string filename, int depth)
{
    filename.append("mandelbrotparallel.txt");
    std::ofstream file(filename);

    for(int i=0; i<100; i++)
    {
         Parallel::Mandelbrot::renderFunction(depth);
        file <<  Parallel::Mandelbrot::getTime()*1000 << " ms" << std::endl;
    }

    file.close();
}

void Tests::runDepthTest()
{
    int depth;

    std::string filename;

    for( depth = 5; depth <=15 ; depth+=5)
    {
        filename = "depth";
        filename.append(std::to_string(depth));

        testTriangleS(filename, depth);
        testTriangleP(filename, depth);
    }
    
    for(depth = 50; depth <=200; depth*=2)
    {
        filename = "depth";
        filename.append(std::to_string(depth));

        testMandelbrotS(filename, depth);
        testMandelbrotP(filename, depth);
    }

}

void Tests::runResTest()
{
    int original = GLManager::getWidth();

    std::string filename;

    for(int res = 500; res <= 2000; res*=2 )
    {
        filename = "res";
        filename.append(std::to_string(res));

        GLManager::setResolution(res);
        testTriangleS(filename, 10);
        testTriangleP(filename, 15);
        testMandelbrotS(filename, 50);
        testMandelbrotP(filename, 500);
    }

    GLManager::setResolution(original);
}

void Tests::runBlockTest()
{
    int originaltri = 6;
    int originalman = 512;

    std::string filename;

    for(int size = 1; size<=1024; size*=2)
    {
        filename= "block";
        filename.append(std::to_string(size));

        testTriangleP(filename, 15);
        testMandelbrotP(filename, 500);
    }

    STrianglePar::setBlockSize(originaltri);
    Parallel::Mandelbrot::setBlockSize(originalman);
}

void Tests::runAllTests()
{
    runDepthTest();
    runResTest();
    runBlockTest();
}
