#pragma once

#include <string>

namespace CudaFractals
{
class Utility
{

private:
    bool isWindows;

    std::string getOsName();

public:
    Utility();
    ~Utility() = default;

    void clear() const;

};
}  // namespace CudaFractals