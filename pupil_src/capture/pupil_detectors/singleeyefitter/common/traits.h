#ifndef singleeyefitter_traits_h__
#define singleeyefitter_traits_h__


#include <ceres/jet.h>

namespace singleeyefitter {

    struct scalar_tag {};
    struct ceres_jet_tag {};

    template<typename T, typename Enabled = void>
    struct ad_traits;

    template<typename T>
    struct ad_traits<T, typename std::enable_if< std::is_arithmetic<T>::value >::type > {
        typedef scalar_tag ad_tag;
        typedef T scalar;
        static inline scalar value(const T& x) { return x; }
    };

    template<typename T, int N>
    struct ad_traits<::ceres::Jet<T, N>> {
        typedef ceres_jet_tag ad_tag;
        typedef T scalar;
        static inline scalar get(const ::ceres::Jet<T, N>& x) { return x.a; }
    };

    template<typename T>
    struct ad_traits < T, typename std::enable_if < !std::is_same<T, typename std::decay<T>::type>::value >::type >
        : public ad_traits<typename std::decay<T>::type> {
    };
}


#endif //singleeyefitter_traits_h__
