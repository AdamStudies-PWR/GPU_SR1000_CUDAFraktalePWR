#include "interface.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include "cudaCheck.hpp"
#include "GLManager.hpp"
#include "STriangleSeq.hpp"
#include "mandelbrotSequential.hpp"
#include "mandelbrotParallel.hpp"

using namespace CudaFractals;

Interface::Interface(void(*seqMandelbrot)(void), 
    void(*parMandelbrot)(void), 
    void(*seqSCarpet)(void), 
    void(*parSCarpet)(void)) : seqMandelbrotDisplay(seqMandelbrot),
                               parMandelbrotDisplay(parMandelbrot),
                               seqSTrinagleDisplay(seqSCarpet),
                               parSTriangleDisplay(parSCarpet) {}

void Interface::printCredits() const 
{
    utils.clear();
    static const char* message = "Authors:\n"
                                 "Adam Krizar 241276\n"
                                 "Katarzyna Czajkowska\n"
                                 "Przemyslaw Mikluszka\n"
                                 "Patryk Skowronski 237454\n"
                                 "Marcin Czepiela\n"
                                 "\nPress any key to continue...";
    std::cout << message;
    getchar();
}

bool Interface::detectGPU() const 
{
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    return devCount > 0;
}

void Interface::mainMenu(int* argc, char** argv) const
{
    double result = 0;
    static std::string menu = "";
    menu.append("------------CUDA Fractals------------\n");
    menu.append("[1] Mandelbrot set - Sequential\n");
    menu.append("[2] Mandelbrot set - Parallel\n");
    menu.append("[3] Sierpinski triangle - Sequential\n");
    menu.append("[4] Sierpinski triangle - Parallel\n");
    menu.append("[5] Authors\n");
    menu.append("[6] Exit\n");
    menu.append(">: ");

    int itrInput = 0;

    while (true) 
    {   
        utils.clear();     
        std::cout << menu;
        switch (getchar()) 
        {
        case '1':
            if (seqMandelbrotDisplay == nullptr)
            {
                std::cout << "Drawing callback not assigned\n";
                getchar();
            }
            else
                std::cout << "Set length: ";
                std::cin >> itrInput;
                result = MandelbrotSequential::generateFractal(itrInput);
                GLManager::GLInitialize(argc, argv, seqMandelbrotDisplay);
                std::cout << "Duration: " << result << " s";
                getchar();  
            break;
        case '2':
            if (parMandelbrotDisplay == nullptr)
            {
                std::cout << "Drawing callback not assigned\n";
                getchar();
            }
            else
                std::cout << "Set length: ";
                std::cin >> itrInput;
                Parallel::Mandelbrot::renderFunction(itrInput);
                GLManager::GLInitialize(argc, argv, parMandelbrotDisplay);
            break;
        case '3':
            if (seqSTrinagleDisplay == nullptr)
            {
                std::cout << "Drawing callback not assigned\n";
                getchar();
            }
            else
                std::cout << "Iterations: ";
                std::cin >> itrInput;
                STriangleSeq::Generate(itrInput);
                std::cout << "Duration: "<<STriangleSeq::GetTime()<<" s";
                GLManager::GLInitialize(argc, argv, seqSTrinagleDisplay);
                getchar();             
            break;
        case '4':
            if (parSTriangleDisplay == nullptr)
            {
                std::cout << "Drawing callback not assigned\n";
                getchar();
            }
            else
                GLManager::GLInitialize(argc, argv, parSTriangleDisplay);
            break;
        case '5':
            printCredits();
            break;
        case '6':
            return;
            break;
        }
        std::cin.ignore();
    }
}
