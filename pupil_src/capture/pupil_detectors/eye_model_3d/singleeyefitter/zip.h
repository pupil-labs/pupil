#ifndef zip_h__
#define zip_h__

#include <boost/config.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>

namespace singleeyefitter {

    template <typename Tuple>
    struct iterator_tuple_traits
    {
        typedef declval(boost::fusion::for_each((*(Tuple*) 0), deref)) Reference;
    };

    template<typename Tuple>
    class zip_iterator : boost::iterator_facade<typename zip_iterator<Tuple>, {
    public:
        template<typename OtherTuple>
        zip_iterator(OtherTuple&& iterators, typename std::enable_if<std::is_convertible<OtherTuple, Tuple>::value>::type* = 0)
            : iterators(std::forward<OtherTuple>(iterators)) {
        }

        Tuple& operator*() {
            return iterators;
        }
        const Tuple& operator*() const {
            return iterators;
        }
        Tuple& operator*() {
            return iterators;
        }
        const Tuple& operator*() const {
            return iterators;
        }
    private:
        Tuple iterators;
    };

#ifndef BOOST_NO_VARIADIC_TEMPLATES

    template <typename... T>
    auto zip(const T&... containers) -> boost::iterator_range < boost::zip_iterator<decltype(boost::make_tuple(std::begin(containers)...))> >
    {
        auto zip_begin = boost::make_zip_iterator(boost::make_tuple(std::begin(containers)...));
        auto zip_end = boost::make_zip_iterator(boost::make_tuple(std::end(containers)...));
        return boost::make_iterator_range(zip_begin, zip_end);
    }

#else

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iteration/local.hpp>

#define ZIP_begin(z, n, data) std::begin(BOOST_PP_CAT(data, n))
#define ZIP_end(z, n, data) std::end(BOOST_PP_CAT(data, n))

#define BOOST_PP_LOCAL_MACRO(n) \
    template <BOOST_PP_ENUM_PARAMS(n, typename T)> \
    auto zip(BOOST_PP_ENUM_BINARY_PARAMS(n, const T, & container)) -> \
            boost::iterator_range<boost::zip_iterator<decltype(std::make_tuple(BOOST_PP_ENUM(n, ZIP_begin, container)))>> \
        { \
        auto zip_begin = boost::make_zip_iterator(std::make_tuple(BOOST_PP_ENUM(n, ZIP_begin, container))); \
        auto zip_end = boost::make_zip_iterator(std::make_tuple(BOOST_PP_ENUM(n, ZIP_end, container))); \
        return boost::make_iterator_range(zip_begin, zip_end); \
        }

#define BOOST_PP_LOCAL_LIMITS (2, 10)
#include BOOST_PP_LOCAL_ITERATE()

#undef ZIP_end
#undef ZIP_begin

#endif

}

#endif // zip_h__
