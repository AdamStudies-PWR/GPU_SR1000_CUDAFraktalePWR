#include "tests.hpp"
#include "STriangleSeq.hpp"
#include "STrianglePar.hpp"
#include "mandelbrotSequential.hpp"
#include "mandelbrotParallel.hpp"

using namespace CudaFractals;

void Tests::testTriangleS(int depth)
{
    std::string fname = "triangleseq";
    fname.append(std::to_string(GLManager::getWidth()));
    fname.append("r");
    fname.append(std::to_string(depth));
    fname.append("d.txt");

    std::ofstream file(fname);

    for(int i=0; i<100; i++)
    {
        STriangleSeq::Generate(depth);
        file << STriangleSeq::GetTime()*1000 << " ms" << std::endl;
    }

    file.close();
}

void Tests::testTriangleP(int depth)
{
    std::string fname = "trianglepar";
    fname.append(std::to_string(GLManager::getWidth()));
    fname.append("r");
    fname.append(std::to_string(depth));
    fname.append("d");
    fname.append(".txt");

    std::ofstream file(fname);

    for(int i=0; i<100; i++)
    {
        STrianglePar::Generate(depth);
        file << STrianglePar::GetTime()*1000 << " ms" << std::endl;
    }

    file.close();
}

void Tests::testMandelbrotS(int depth)
{
    std::string fname = "mandelbrotseq";
    fname.append(std::to_string(GLManager::getWidth()));
    fname.append("r");
    fname.append(std::to_string(depth));
    fname.append("d.txt");

    std::ofstream file(fname);

    for(int i=0; i<100; i++)
    {
        file << MandelbrotSequential::generateFractal(depth)*1000 << " ms" << std::endl;
    }

    file.close();
}

void Tests::testMandelbrotP(int depth)
{
    std::string fname = "mandelbrotpar";
    fname.append(std::to_string(GLManager::getWidth()));
    fname.append("r");
    fname.append(std::to_string(depth));
    fname.append("d");
    fname.append(".txt");

    std::ofstream file(fname);

    for(int i=0; i<100; i++)
    {
         Parallel::Mandelbrot::renderFunction(depth);
        file <<  Parallel::Mandelbrot::getTime()*1000 << " ms" << std::endl;
    }

    file.close();
}

void Tests::runAllTests(int res, int originalRes)
{
    GLManager::setResolution(res);

    int depth;

    for( depth = 5; depth <=15 ; depth+=5)
    {
        testTriangleS(depth);
        testTriangleP(depth);
    }
    
    for(depth = 50; depth <=200; depth*=2)
    {
        testMandelbrotS(depth);
        testMandelbrotP(depth);
    }

    GLManager::setResolution(originalRes);
}
