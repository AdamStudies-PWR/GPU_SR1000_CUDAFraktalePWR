#include "utility.hpp"

using namespace CudaFractals;

Utility::Utility()
{
    if(getOsName() == "WIN") isWindows = true;
    else isWindows = false;
}

std::string Utility::getOsName()
{
    #ifdef _WIN32
    return "WIN";
    #elif _WIN64
    return "WIN";
    #else
    return "LIN";
    #endif
}

void Utility::clear() const
{
    if(isWindows) system("cls");
    else system("clear");
}