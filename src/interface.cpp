#include "interface.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include "cudaCheck.hpp"
#include "GLManager.hpp"
#include "STriangleSeq.hpp"
#include "STrianglePar.hpp"
#include "mandelbrotSequential.hpp"
#include "mandelbrotParallel.hpp"
#include "tests.hpp"

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
                                 "Katarzyna Czajkowska 242079\n"
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
    menu.append("[5] Settings\n");
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
                std::cout << "Duration: " << result << " s";
                GLManager::GLInitialize(argc, argv, seqMandelbrotDisplay);
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
                std::cout << "Duration: "<< Parallel::Mandelbrot::getTime()<<" s";
                GLManager::GLInitialize(argc, argv, parMandelbrotDisplay);
                getchar();
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
                std::cout << "Iterations: ";
                std::cin >> itrInput;
                STrianglePar::SetupDrawingColor(0.5f, 0.5f, 1.0f);
                STrianglePar::Generate(itrInput);
                std::cout << "Duration: "<<STrianglePar::GetTime()<<" s";
                GLManager::GLInitialize(argc, argv, parSTriangleDisplay);
                getchar();  
            break;
        case '5':
            settingsMenu();
            break;
        case '6':
            return;
            break;
        case '8':
            std::cout << "... hacking in progress ... \n";
            getchar();
            testingMenu();
            break;
        }
        std::cin.ignore();
    }
}

void Interface::settingsMenu() const
{
    std::cin.ignore();

    static std::string menu = "";
    menu.append("------------Settings------------\n");
    menu.append("[1] Authors\n");
    menu.append("[2] Change window resolution\n");
    menu.append("[3] Change Mandelbrot set block size\n");
    menu.append("[4] Change Siepinski triangle block size\n");
    menu.append("[5] Enter testing menu \n");
    menu.append("[6] Previous menu\n");
    menu.append(">: ");

    int itrInput = 0;

    while (true) 
    {   
        utils.clear();     
        std::cout << menu;
        switch (getchar()) 
        {
            case '1':
                printCredits();
                break;
            case '2':
                std::cout << "Resolution (width): ";
                std::cin >> itrInput;
                GLManager::setResolution(itrInput);
                break;
            case '3':
                std::cout << "Block size: ";
                std::cin >> itrInput;
                Parallel::Mandelbrot::setBlockSize(itrInput);
                break;
            case '4':
                std::cout << "Block size: ";
                std::cin >> itrInput;
                STrianglePar::setBlockSize(itrInput);
                break;
            case '5':
                testingMenu();
                break;
            case '6':
                return;
                break;
        }
        std::cin.ignore();
    }
}

void Interface::testingMenu() const
{
    std::cin.ignore();

    static std::string menu = "";
    menu.append("------------Testing------------\n");
    menu.append("[1] All tests.\n");
    menu.append("[2] Depth test.\n");
    menu.append("[3] Resolution test.\n");
    menu.append("[4] Block size test. \n");
    menu.append("[5] \n");
    menu.append("[6] Previous menu\n");
    menu.append(">: ");

    int itrInput = 0;

    while (true) 
    {   
        utils.clear();     
        std::cout << menu;
        switch (getchar()) 
        {
            case '1':
                Tests::runAllTests();
                break;
            case '2':
                Tests::runDepthTest();
                break;
            case '3':
                Tests::runResTest();
                break;
            case '4':
                Tests::runBlockTest();
                break;
            case '5':
//                std::cout <<"Depth: ";
//                std::cin >> itrInput;
//                Tests::testMandelbrotP(itrInput);
                break;
            case '6':
                return;
                break;
        }
        std::cin.ignore();
    }
}
