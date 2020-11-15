#include "GLManager.hpp"
#include <vector>
#include <ctime>
#include <ratio>
#include <chrono>

class STriangleSeq {

public:
	struct Vertex2D {
		GLfloat x;
		GLfloat y;	
	};

	struct Triangle {
		Vertex2D p0;
		Vertex2D p1;
		Vertex2D p2;
	};

private:
	
	static double time;
	static std::vector<Triangle> triangles;
public:
	static void Generate(int iterations);
	static void DrawTriangleList();
	static double GetTime();
	
};
