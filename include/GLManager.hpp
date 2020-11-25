#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif
#include <GL/gl.h>  // Includy zmienione do Linxa
#include <GL/freeglut.h>  // Tak�e zmiana do Linuxa, je�eli nie dzia�a na windzie to tylko wykomentowa�, nie usuwa� 
#include <vector>

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

class GLManager 
{
private:
	static void ChangeSize(GLsizei horizontal, GLsizei vertical);
public:
	static void GLInitialize(int* argc, char** argv, void(*displayCallback)(void));

	static int getWidth();
	static int getHeight();
};
