#include "STriangleSeq.hpp"

std::vector<STriangleSeq::Triangle> STriangleSeq::triangles;
double STriangleSeq::time;

void STriangleSeq::Generate(int iterations) {

	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point stop;
	int width = GLManager::getWidth();
	int height = GLManager::getHeight();
	Vertex2D p0 = { -width/2, -height/2 };
	Vertex2D p1 = { width/2, -height/2 };
	Vertex2D p2 = { 0.0f, height/2 };
	Vertex2D p3, p4, p5;
	Triangle triangle = { p0, p1, p2 };
	std::vector < STriangleSeq::Triangle > iterationBuffer;
	
	iterationBuffer.push_back(triangle);	
	time = 0;

	for (int i = 1; i <= iterations; i++) {
		triangles.clear();
		triangles.reserve(pow(3, (double)i));
		
		start = std::chrono::high_resolution_clock::now();
		for (auto& tri : iterationBuffer) {
	
			p3.x = tri.p0.x + (abs(tri.p1.x - tri.p0.x) / 2.0f);
			p3.y = tri.p0.y;

			p5.x = tri.p0.x + (abs(tri.p2.x - tri.p0.x) / 2.0f);
			p5.y = tri.p0.y + (abs(tri.p2.y - tri.p0.y) / 2.0f);

			p4.x = p3.x + (abs(tri.p1.x - tri.p2.x) / 2.0f);
			p4.y = p3.y + (abs(tri.p1.y - tri.p2.y) / 2.0f);

			triangle = { tri.p0,p3,p5 };
			triangles.push_back(triangle);

			triangle = { p3,tri.p1,p4 };
			triangles.push_back(triangle);

			triangle = { p5,p4,tri.p2 };
			triangles.push_back(triangle);	
		}
		stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
		time += duration.count();
		iterationBuffer = triangles;
	}
}

void STriangleSeq::DrawTriangleList() {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.5f, 1.0f, 0.5f);
	for (auto& triangle : triangles) {	
		glBegin(GL_TRIANGLES);
		glVertex2f(triangle.p0.x, triangle.p0.y);
		glVertex2f(triangle.p1.x, triangle.p1.y);
		glVertex2f(triangle.p2.x, triangle.p2.y);
		glEnd();
	}
	glFlush();
}

double STriangleSeq::GetTime() {
	return time;
}
