#include <iostream>
#include "interface.hpp"
#include "GLManager.hpp"
#include "STriangleSeq.hpp"
#include "mandelbrotSequential.hpp"

// Przykładowa funkcja rysujaca
void RenderScene(void){
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush();    
}

void mandelbrotSequential(void)
{
    CudaFractals::MandelbrotSequential ms;
    ms.generateFractal();    
}

int main(int argc, char* argv[])
{
    
    // Tutaj podajemy wskaźniki na pisane przez nas funkcje rysujące odpowiednie fraktale

    CudaFractals::Interface interf(mandelbrotSequential, nullptr, STriangleSeq::DrawTriangleList, nullptr);
    std::cout << "Checking GPU..." << std::endl;

    if (interf.detectGPU()) {
        std::cout << "\nCheck ok. Press any key to continue...\n";
        getchar();
        interf.mainMenu(&argc,argv);
    }
    else
        std::cout << "\nNo CUDA enabled device detected\n";

    return EXIT_SUCCESS;
}