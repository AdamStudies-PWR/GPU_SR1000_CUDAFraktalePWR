#pragma once

#include "constexprMath.hpp"

// skalowanie wspolrzednych - mandelbrot
constexpr float xScaleMandelbrotStart = -2.5;
constexpr float xScaleMandelbrotEnd = 1;
constexpr float yScaleMandelbrotStart = -1;
constexpr float yScaleMandelbrotEnd = 1;
constexpr float xScaleMandelbrotWidth =
    ConstexprMath::abs(xScaleMandelbrotEnd) +
    ConstexprMath::abs(xScaleMandelbrotStart);
constexpr float yScaleMandelbrotWidth =
    ConstexprMath::abs(yScaleMandelbrotEnd) +
    ConstexprMath::abs(yScaleMandelbrotStart);
