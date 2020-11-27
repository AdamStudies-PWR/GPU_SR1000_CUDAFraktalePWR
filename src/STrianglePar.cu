#include <cuda_runtime.h>
#include <chrono>

#include "STrianglePar.hpp"
//#include "cudaCheck.hpp"

std::vector<STrianglePar::Triangle> STrianglePar::triangles;
GLfloat STrianglePar::r, STrianglePar::g, STrianglePar::b;
double STrianglePar::time;
int STrianglePar::blockSize = 6;

__global__ void SubdivideTriangles(STrianglePar::Triangle* sourceTriangles, STrianglePar::Triangle* subdividedTriangles) {
	STrianglePar::Vertex2D p3, p4, p5;
	STrianglePar::Triangle triangle{0,0,0};
	long long baseIndex = 3 * blockIdx.x;
	
	p3.x = sourceTriangles[blockIdx.x].p0.x + (abs(sourceTriangles[blockIdx.x].p1.x - sourceTriangles[blockIdx.x].p0.x) / 2.0f);
	p3.y = sourceTriangles[blockIdx.x].p0.y;

	p5.x = sourceTriangles[blockIdx.x].p0.x + (abs(sourceTriangles[blockIdx.x].p2.x - sourceTriangles[blockIdx.x].p0.x) / 2.0f);
	p5.y = sourceTriangles[blockIdx.x].p0.y + (abs(sourceTriangles[blockIdx.x].p2.y - sourceTriangles[blockIdx.x].p0.y) / 2.0f);

	p4.x = p3.x + (abs(sourceTriangles[blockIdx.x].p1.x - sourceTriangles[blockIdx.x].p2.x) / 2.0f);
	p4.y = p3.y + (abs(sourceTriangles[blockIdx.x].p1.y - sourceTriangles[blockIdx.x].p2.y) / 2.0f);

	triangle = { sourceTriangles[blockIdx.x].p0,p3,p5 };

	subdividedTriangles[baseIndex] = triangle;
	
	triangle = { p3,sourceTriangles[blockIdx.x].p1,p4 };
	subdividedTriangles[baseIndex+1] = triangle;

	triangle = { p5,p4,sourceTriangles[blockIdx.x].p2 };
	subdividedTriangles[baseIndex+2] = triangle;
}

void STrianglePar::SetupDrawingColor(GLfloat r_, GLfloat g_, GLfloat b_) {
	r = r_;
	g = g_;
	b = b_;
}

void STrianglePar::Generate(int iterations) {

	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point stop;

	int width = GLManager::getWidth();
	int height = GLManager::getHeight();
	Vertex2D p0 = { -width/2, -height/2 };
	Vertex2D p1 = { width/2, -height/2 };
	Vertex2D p2 = { 0.0f, height/2 };
	Triangle* sourceTrisOnDevice;
	Triangle* subdividedTrisOnDevice;
	Triangle triangle = { p0, p1, p2 };
	long long subdividedTrisVecSize;
	long long numBlocks;

	triangles.resize(pow(3, (double)iterations));
	time = 0;
	
	start = std::chrono::high_resolution_clock::now();
	
	cudaMalloc((void**)&sourceTrisOnDevice, sizeof(Triangle));
	cudaMemcpy(sourceTrisOnDevice, &triangle, sizeof(Triangle), cudaMemcpyHostToDevice);
	
	for (int i = 1; i <= iterations; i++) {
	
		numBlocks = pow(3, (double)(i - 1));	
		subdividedTrisVecSize = pow(3, (double)i) * sizeof(Triangle);
		
		cudaMalloc((void**)&subdividedTrisOnDevice, subdividedTrisVecSize);
	
		SubdivideTriangles<<<numBlocks,blockSize>>>(sourceTrisOnDevice, subdividedTrisOnDevice);
		cudaDeviceSynchronize();
		cudaFree(sourceTrisOnDevice);
		sourceTrisOnDevice = subdividedTrisOnDevice;		
	}
	
	cudaMemcpy(triangles.data(), sourceTrisOnDevice, subdividedTrisVecSize, cudaMemcpyDeviceToHost);
	cudaFree(subdividedTrisOnDevice);

	stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	time = duration.count();	
}

void STrianglePar::DrawTriangleList() {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(r, g, b);
	for (auto& triangle : triangles) {	
		glBegin(GL_TRIANGLES);
		glVertex2f(triangle.p0.x, triangle.p0.y);
		glVertex2f(triangle.p1.x, triangle.p1.y);
		glVertex2f(triangle.p2.x, triangle.p2.y);
		glEnd();
	}
	glFlush();
}

double STrianglePar::GetTime() {
	return time;
}

void STrianglePar::setBlockSize(int size)
{
  if(size > 0) blockSize = size;
}
