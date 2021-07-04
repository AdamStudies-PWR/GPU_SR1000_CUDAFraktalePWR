#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif
#include <GL/gl.h>  // Includy zmienione do Linxa
#include <GL/freeglut.h>  // Tak�e zmiana do Linuxa, je�eli nie dzia�a na windzie to tylko wykomentowa�, nie usuwa� 
#include <vector>

class GLManager 
{
private:
	static int window_width;
	static int window_height;

	static void ChangeSize(GLsizei horizontal, GLsizei vertical);
public:
	static void GLInitialize(int* argc, char** argv, void(*displayCallback)(void));

	static int getWidth();
	static int getHeight();

	static void setResolution(int);
};
