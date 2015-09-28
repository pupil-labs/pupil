#ifndef singleeyefitter_math_h__
#define singleeyefitter_math_h__

#include <limits>

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

    }

}


#endif // singleeyefitter_math_h__
