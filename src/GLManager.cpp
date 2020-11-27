#include "interface.hpp"
#include "GLManager.hpp"

	int GLManager::window_width = 1280;
	int GLManager::window_height = 720;

void GLManager::ChangeSize(GLsizei horizontal, GLsizei vertical)
{
    glMatrixMode(GL_PROJECTION);
    glOrtho(-horizontal/2, horizontal/2, -vertical/2, vertical/2, 10.0, -10.0);
    glMatrixMode(GL_MODELVIEW);                     
	glLoadIdentity();
}

void GLManager::GLInitialize(int* argc, char** argv, void(*displayCallback)(void)) {   
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA Fractals");
    glutDisplayFunc(displayCallback);
    glutReshapeFunc(ChangeSize);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, window_width, window_height);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
              GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutMainLoop();
}

int GLManager::getHeight()
{
    return window_height;
}


int GLManager::getWidth()
{
    return window_width;
}

void GLManager::setResolution(int width)
{
    if(width <= 150) width = 1280;

    window_width = width;
    window_height = (9.0/16.0)*width;
}