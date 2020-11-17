#include "mandelbrotSequential.hpp"
#include "GLManager.hpp"
#include "iostream"

using namespace CudaFractals;

std::vector<MandelbrotSequential::Colors> MandelbrotSequential::colors;

int MandelbrotSequential::height;
int MandelbrotSequential::width;
int MandelbrotSequential::LENGTH;

double MandelbrotSequential::generateFractal(int length)
{
    if(length <= 0) LENGTH = 100;
    else LENGTH = length;
    std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point stop;
    start = std::chrono::high_resolution_clock::now();

    height = GLManager::getHeight();
    width = GLManager::getWidth();
    int counter = 0;

    Colors pixel;
    colors.clear();
    colors.reserve(width * height);

    float x, y, scale, r, g, b;

    for (int i=-(width/2); i<width/2; i++)
    {
        for (int j=-(height/2); j<height/2; j++)
        {
            //Przusnięcie o width/4 żeby wyśrodkować w oknie
            x = (float)(i - width/4)/(width/3.5);
            y = (float)j/(height/2);
            std::complex<float> C(x, y);
            scale = mandelbrotPoint(C) * 0.02;
            r = (0.5 - scale/2);
            r = r < 0 ? 0 : r;
            g = (1.0 - scale);
            g = g < 0 ? 0 : g;
            b = (0.5 - scale/2);
            b = b < 0 ? 0 : b;
            pixel = {r, g, b};
            colors.push_back(pixel);
            counter++;
        }
    }  

    stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    return duration.count();
}

void MandelbrotSequential::draw()
{
    int counter = 0;
    float scale;
    float r, g, b;

    glClear(GL_COLOR_BUFFER_BIT);
        glPointSize(1.0f);
        glBegin(GL_POINTS);
            for (int i=-(width/2); i<width/2; i++)
            {
                for (int j=-(height/2); j<height/2; j++)
                {
                    glColor3f(colors[counter].r, colors[counter].g, colors[counter].b);
                    glVertex2i(i, j);
                    counter++;
                }
            }                          
        glEnd();
    glFlush();
}

int MandelbrotSequential::mandelbrotPoint(std::complex<float> C)
{  
    std::complex<float> Z(0, 0);
    int counter = 0;
    while (abs(Z) <= 2 && counter < LENGTH)
    {
        Z = Z*Z + C;
        counter++;
    }
    return counter;
}
