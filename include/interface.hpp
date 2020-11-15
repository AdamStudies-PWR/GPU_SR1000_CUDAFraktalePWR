#pragma once

#include "utility.hpp"

namespace CudaFractals 
{

class Interface 
{

private:
    void(*seqMandelbrotDisplay)(void);
    void(*parMandelbrotDisplay)(void);
    void(*seqSTrinagleDisplay)(void);
    void(*parSTriangleDisplay)(void);
    CudaFractals::Utility utils;

public:
    Interface(void(*seqManedlbrot)(void),
        void(*parManedlbrot)(void),
        void(*seqSCarpet)(void),
        void(*parSCarpet)(void));
    ~Interface() = default;
    void printCredits() const;
    bool detectGPU() const;
    void mainMenu(int* argc, char** argv) const;

};
}  // namespace CudaFractals
