#include "GLManager.hpp"
#include <vector>

class STrianglePar
{

public:
    struct Vertex2D
    {
        GLfloat x;
        GLfloat y;
    };

    struct Triangle
    {
        Vertex2D p0;
        Vertex2D p1;
        Vertex2D p2;
    };

private:
    static double time;
	static std::vector<Triangle> triangles;
	static GLfloat r, g, b;

public:
    static void SetupDrawingColor(GLfloat r_, GLfloat g_, GLfloat b_);
    static void Generate(int iterations);
    static void DrawTriangleList();
    static double GetTime();
};
