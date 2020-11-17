#pragma once
// Windows
// #include <windows.h>  // do zakomentowania na linuxie ofc
// #include <gl/GL.h>  // do zakomentowania na linuxie ofc
// #include <gl/freeglut.h>  // do zakomentowania na linuxie ofc 
// Linux
#include <GL/gl.h>  // Includy zmienione do Linxa
#include <GL/freeglut.h>  // Tak�e zmiana do Linuxa, je�eli nie dzia�a na windzie to tylko wykomentowa�, nie usuwa� 
#include <vector>

class GLManager 
{
private:
	static void ChangeSize(GLsizei horizontal, GLsizei vertical);
public:
	static void GLInitialize(int* argc, char** argv, void(*displayCallback)(void));

	static int getWidth();
	static int getHeight();
};
