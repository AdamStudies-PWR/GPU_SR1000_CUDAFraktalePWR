cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FractalApp LANGUAGES CXX CUDA)

# CUDA
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# GLUT
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

include_directories("include/")
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

add_executable(FractalApp main.cu ${SOURCES} 
    "include/GLManager.hpp" "src/GLManager.cpp" 
    "include/STriangleSeq.hpp" "src/STriangleSeq.cpp" 
    "include/mandelbrotSequential.hpp" 
    "src/mandelbrotSequential.cpp" 
    "include/mandelbrotParallel.hpp" 
    "src/mandelbrotParallel.cu"
    "include/STrianglePar.hpp"
    "src/STrianglePar.cu"
    "include/tests.hpp"
)

target_compile_features(FractalApp PUBLIC cxx_std_17)
target_link_libraries(FractalApp ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} GLUT::GLUT)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
