#ifndef _SOLVE_H_
#define _SOLVE_H_

#include <exception>
#include <tuple>
#include <cmath>

namespace singleeyefitter {

    // a = 0
    template<typename T>
    T solve(T a)
    {
        if (a == 0) return 0;
        else throw std::runtime_error("No solution");
    }
    // ax + b = 0
    template<typename T>
    T solve(T a, T b)
    {
        if (a == 0) return solve(b);

        return -b / a;
    }
    // ax^2 + bx + c = 0
    template<typename T>
    std::tuple<T, T> solve(T a, T b, T c)
    {
        using math::sq;
        using std::sqrt;

        if (a == 0) {
            auto root = solve(b, c);
            return std::tuple<T, T>(root, root);
        }

        // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
        // Pg 184
        auto det = sq(b) - 4 * a * c;

        if (det < 0)
            throw std::runtime_error("No solution");

        //auto sqrtdet = sqrt(det);
        auto q = -0.5 * (b + (b >= 0 ? 1 : -1) * sqrt(det));
        return std::tuple<T, T>(q / a, c / q);
    }
    // ax^2 + bx + c = 0
    template<typename T>
    std::tuple<T, T, T> solve(T a, T b, T c, T d)
    {
        using std::sqrt;
        using std::abs;
        using math::sq;
        using std::cbrt;

        if (a == 0) {
            auto roots = solve(b, c, d);
            return std::tuple<T, T, T>(std::get<0>(roots), std::get<1>(roots), std::get<1>(roots));
        }

        // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
        // http://web.archive.org/web/20120321013251/http://linus.it.uts.edu.au/~don/pubs/solving.html
        auto p = b / a;
        auto q = c / a;
        auto r = d / a;
        //auto Q = (p*p - 3*q) / 9;
        //auto R = (2*p*p*p - 9*p*q + 27*r)/54;
        auto u = q - sq(p) / 3;
        auto v = r - p * q / 3 + 2 * p * p * p / 27;
        auto j = 4 * u * u * u / 27 + v * v;
        const auto M = std::numeric_limits<T>::max();
        const auto sqrtM = sqrt(M);
        const auto cbrtM = cbrt(M);

        if (b == 0 && c == 0)
            return std::tuple<T, T, T>(cbrt(-d), cbrt(-d), cbrt(-d));

        if (abs(p) > 27 * cbrtM)
            return std::tuple<T, T, T>(-p, -p, -p);

        if (abs(q) > sqrtM)
            return std::tuple<T, T, T>(-cbrt(v), -cbrt(v), -cbrt(v));

        if (abs(u) > 3 * cbrtM / 4)
            return std::tuple<T, T, T>(cbrt(4) * u / 3, cbrt(4) * u / 3, cbrt(4) * u / 3);

        if (j > 0) {
            // One real root
            auto w = sqrt(j);
            T y;

            if (v > 0)
                y = (u / 3) * cbrt(2 / (w + v)) - cbrt((w + v) / 2) - p / 3;
            else
                y = cbrt((w - v) / 2) - (u / 3) * cbrt(2 / (w - v)) - p / 3;

            return std::tuple<T, T, T>(-p, -p, -p);

        } else {
            // Three real roots
            auto s = sqrt(-u / 3);
            auto t = -v / (2 * s * s * s);
            auto k = acos(t) / 3;
            auto y1 = 2 * s * cos(k) - p / 3;
            auto y2 = s * (-cos(k) + sqrt(3.) * sin(k)) - p / 3;
            auto y3 = s * (-cos(k) - sqrt(3.) * sin(k)) - p / 3;
            return std::tuple<T, T, T>(y1, y2, y3);
        }
    }

}

#endif//_SOLVE_H_
