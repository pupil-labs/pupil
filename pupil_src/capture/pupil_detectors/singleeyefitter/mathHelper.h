#ifndef singleeyefitter_math_h__
#define singleeyefitter_math_h__

#include <limits>

#include "traits.h"

namespace singleeyefitter {

    namespace math {

#define MAKE_SQ(TYPE) \
    inline auto sq(TYPE val) -> decltype(val*val) { return val * val; }

        MAKE_SQ(float)
            MAKE_SQ(double)
            MAKE_SQ(long double)
            MAKE_SQ(char)
            MAKE_SQ(short)
            MAKE_SQ(int)
            MAKE_SQ(long)
            MAKE_SQ(long long)
            MAKE_SQ(unsigned char)
            MAKE_SQ(unsigned short)
            MAKE_SQ(unsigned int)
            MAKE_SQ(unsigned long)
            MAKE_SQ(unsigned long long)

#undef MAKE_STD_SQ

        template<typename T, typename TMin, typename TMax>
        inline T clamp(T val, TMin min_val = std::numeric_limits<T>::min(), TMax max_val = std::numeric_limits<T>::max()) {
            if (min_val > max_val)
                return clamp(val, max_val, min_val);
            if (val <= min_val)
                return min_val;
            if (val >= max_val)
                return max_val;
            return val;
        }

        template<typename T>
        inline T lerp(const T& val1, const T& val2, const T& alpha)
        {
            return val1*(1.0 - alpha) + val2*alpha;
        }

        template<typename T>
        float getAngleABC( const T& a, const T& b, const T& c )
        {
            T ab = { b.x - a.x, b.y - a.y };
            T cb = { b.x - c.x, b.y - c.y };

            float dot = ab.dot(cb); // dot product
            float cross = ab.cross(cb); // cross product

            float alpha = atan2(cross, dot);

            return alpha * 180.0f / M_PI;
        }

        template<typename T>
        inline T smootherstep(T edge0, T edge1, T x, scalar_tag)
        {
            if (x >= edge1)
                return T(1);
            else if (x <= edge0)
                return T(0);
            else {
                x = (x - edge0)/(edge1 - edge0);
                return x*x*x*(x*(x*T(6) - T(15)) + T(10));
            }
        }
        template<typename T, int N>
        inline ::ceres::Jet<T,N> smootherstep(T edge0, T edge1, const ::ceres::Jet<T,N>& f, ceres_jet_tag)
        {
            if (f.a >= edge1)
                return ::ceres::Jet<T,N>(1);
            else if (f.a <= edge0)
                return ::ceres::Jet<T,N>(0);
            else {
                T x = (f.a - edge0)/(edge1 - edge0);

                // f is referenced by this function, so create new value for return.
                ::ceres::Jet<T,N> g;
                g.a = x*x*x*(x*(x*T(6) - T(15)) + T(10));
                g.v = f.v * (x*x*(x*(x*T(30) - T(60)) + T(30))/(edge1 - edge0));
                return g;
            }
        }
        template<typename T, int N>
        inline ::ceres::Jet<T,N> smootherstep(T edge0, T edge1, ::ceres::Jet<T,N>&& f, ceres_jet_tag)
        {
            if (f.a >= edge1)
                return ::ceres::Jet<T,N>(1);
            else if (f.a <= edge0)
                return ::ceres::Jet<T,N>(0);
            else {
                T x = (f.a - edge0)/(edge1 - edge0);

                // f is moved into this function, so reuse it.
                f.a = x*x*x*(x*(x*T(6) - T(15)) + T(10));
                f.v *= (x*x*(x*(x*T(30) - T(60)) + T(30))/(edge1 - edge0));
                return f;
            }
        }
        template<typename T>
        inline auto smootherstep(typename ad_traits<T>::scalar edge0, typename ad_traits<T>::scalar edge1, T&& val)
            -> decltype(smootherstep(edge0, edge1, std::forward<T>(val), typename ad_traits<T>::ad_tag()))
        {
            return smootherstep(edge0, edge1, std::forward<T>(val), typename ad_traits<T>::ad_tag());
        }

        template<typename T>
        inline T norm(T x, T y, scalar_tag) {
            using std::sqrt;
            using math::sq;

            return sqrt(sq(x) + sq(y));
        }
        template<typename T, int N>
        inline ::ceres::Jet<T,N> norm(const ::ceres::Jet<T,N>& x, const ::ceres::Jet<T,N>& y, ceres_jet_tag) {
            T anorm = norm<T>(x.a, y.a, scalar_tag());

            ::ceres::Jet<T,N> g;
            g.a = anorm;
            g.v = (x.a/anorm)*x.v + (y.a/anorm)*y.v;

            return g;
        }
        template<typename T>
        inline typename std::decay<T>::type norm(T&& x, T&& y) {
            return norm(std::forward<T>(x), std::forward<T>(y), typename ad_traits<T>::ad_tag());
        }

        template<typename T>
        inline auto Heaviside(T&& val, typename ad_traits<T>::scalar epsilon) -> decltype(smootherstep(-epsilon, epsilon, std::forward<T>(val))) {
            return smootherstep(-epsilon, epsilon, std::forward<T>(val));
        }


    } // math namespace

} // singleeyefitter namespace


#endif // singleeyefitter_math_h__
