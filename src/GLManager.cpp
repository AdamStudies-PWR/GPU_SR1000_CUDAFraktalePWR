#include "interface.hpp"
#include "GLManager.hpp"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

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
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("CUDA Fractals");
    glutDisplayFunc(displayCallback);
    glutReshapeFunc(ChangeSize);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
              GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutMainLoop();
}

int GLManager::getHeight()
{
    return glutGet(GLUT_WINDOW_HEIGHT);
}


int GLManager::getWidth()
{
    return glutGet(GLUT_WINDOW_WIDTH);
}
