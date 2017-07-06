#ifndef singleeyefitter_fun_h__
#define singleeyefitter_fun_h__

#include <iterator>
#include <algorithm>
#include <map>
#include <vector>

#include <boost/iterator_adaptors.hpp>

//#define LAMBDA2(...) { return __VA_ARGS__; }
//#define LAMBDA(...) [&](__VA_ARGS__)LAMBDA2

namespace singleeyefitter {

    namespace fun {

        template<class Container, class Start, class End, class Step>
        Container range_(Start start, End end, Step step)
        {
            Container c;
            typedef typename std::decay<decltype(c[0])>::type Value;

            for (Value i = start; i < end; i += step) {
                c.emplace_back(i);
            }

            return c;
        }
        template<class Container, class Start, class End>
        Container range_(Start start, End end)
        {
            return range_<Container, Start, End, int>(start, end, 1);
        }
        template<class Container, class End>
        Container range_(End end)
        {
            return range_<Container, End, End, int>(0, end, 1);
        }

        template<typename T>
        struct LinspaceIterable {
                struct LinspaceIterator : public boost::iterator_adaptor < LinspaceIterator, int, T, boost::forward_traversal_tag, T, int > {
                    T scale, offset;
                    LinspaceIterator(T scale, T offset, int i) : scale(scale), offset(offset), LinspaceIterator::iterator_adaptor_(i) { }
                    friend boost::iterator_core_access;
                    T dereference() const { return this->base_reference() * scale + offset; }
                };

                typedef LinspaceIterator iterator;
                typedef LinspaceIterator const_iterator;

                T start_val, end_val;
                int steps;

                LinspaceIterable(T start, T end, int steps) : start_val(start), end_val(end), steps(steps) {}
                iterator begin() const { return LinspaceIterator((end_val - start_val) / steps, start_val, 0); }
                iterator end() const { return LinspaceIterator((end_val - start_val) / steps, start_val, steps); }
        };

        template<class T>
        LinspaceIterable<T> linspace(T start, T end, int steps)
        {
            return LinspaceIterable<T>(start, end, steps);
        }

        template<class Container, class Start, class End>
        Container linspace_(Start start, End end, int steps)
        {
            Container c;
            typedef decltype(c[0]) Value;

            for (int i = 0; i < steps; ++i) {
                Value value = i * (end - start) / steps + start;
                c.emplace_back(value);
            }

            return c;
        }

        namespace detail {
            template<class Container, class Function>
            struct map_helper;

            template<class T, class Allocator, class Function>
            struct map_helper < std::vector<T, Allocator>, Function > {
                typedef std::vector<decltype(std::declval<Function>()(std::declval<T>()))> return_type;

                static return_type map(const Function& func, const std::vector<T, Allocator>& src)
                {
                    return_type ret;
                    ret.reserve(src.size());

                    for (const auto& x : src) {
                        ret.emplace_back(func(x));
                    }

                    return ret;
                }
            };

            template<class Function, class Key, class T, class Compare, class Allocator>
            struct map_helper < std::map<Key, T, Compare, Allocator>, Function > {
                typedef std::map<Key, decltype(std::declval<Function>()(std::declval<T>()))> return_type;

                static return_type map(const Function& func, const std::map<Key, T, Compare, Allocator>& src)
                {
                    return_type ret;

                    for (const auto& x : src) {
                        ret.emplace(x.first, func(x.second));
                    }

                    return ret;
                }
            };
        }

        template<class Container, class Function>
        typename detail::map_helper<Container, Function>::return_type map(Function func, Container src)
        {
            return detail::map_helper<Container, Function>::map(std::forward<Function>(func), std::forward<Container>(src));
        }


        namespace detail {
            template<class Container, class Function>
            struct filter_helper;

            template<class T, class Allocator, class Function>
            struct filter_helper < std::vector<T, Allocator>, Function > {
                static std::vector<T, Allocator> filter(const Function& func, const std::vector<T, Allocator>& src)
                {
                    std::vector<T, Allocator> ret;
                    ret.reserve(src.size());

                    for (const auto& x : src) {
                        if (func(x))
                            ret.emplace_back(x);
                    }

                    return ret;
                }
                static std::vector<T, Allocator> filter(const Function& func, std::vector<T, Allocator>&& vec)
                {
                    vec.erase(std::remove_if(begin(vec), end(vec), [&func](const T & x) {return !func(x); }), end(vec));
                    return vec;
                }
            };
        }

        template<class Container, class Function>
        Container filter(Function func, Container src)
        {
            return detail::filter_helper<Container, Function>::filter(std::forward<Function>(func), std::forward<Container>(src));
        }


        template< class Container1, class Container2 >
        bool isSubset(Container1& c1, Container2& c2)
        {
            bool is_subset = false;

            for (auto& c : c2) {
                is_subset |= std::includes(begin(c1), end(c1), begin(c), end(c));
            }

            return is_subset;
        }

        template<class Container, class Allocator >
        Container flatten(const std::vector<Container, Allocator>& v)
        {
            std::size_t total_size = 0;
            for (const auto& sub : v)
                total_size += sub.size();

            Container result;
            result.reserve(total_size);

            for (const auto& sub : v)
                result.insert(result.end(), sub.begin(), sub.end());

            return result;
        }

        /*namespace internal {
            template<class Container, class Function>
            struct reduce_helper {
            typedef typename std::result_of<Function>::type acc_type;

            static acc_type reduce(const Function& func, const Container& src, acc_type acc) {
            for (const auto& x : src) {
            acc = func(std::move(acc),x);
            }
            return acc;
            }
            static acc_type reduce(const Function& func, const Container& src) {
            if (src.size == 0)
            return acc_type();

            auto it = begin(src);
            auto last = end(src);
            acc_type acc = *it;
            for (; it < last; ++it) {
            acc = func(acc,x);
            }
            return acc;
            }
            };
            }

            template<class Container, class Function>
            typename internal::reduce_helper<Container, Function>::acc_type reduce(Function func, Container src) {
            return internal::reduce_helper<Container, Function>::reduce(std::forward<Function>(func), std::forward<Container>(src));
            }
            template<class Container, class Function>
            typename internal::reduce_helper<Container, Function>::acc_type reduce(Function func, Container src, typename std::result_of<Function>::type init) {
            return internal::reduce_helper<Container, Function>::reduce(std::forward<Function>(func), std::forward<Container>(src), std::move(init));
            }*/

        template<class Container>
        auto sum(Container src) -> decltype(*begin(src))
        {
            typedef decltype(*begin(src)) sum_type;
            auto it = begin(src);
            auto last = end(src);

            if (it == last)
                return sum_type();

            sum_type acc = *it;
            ++it;

            for (; it < last; ++it) {
                acc += *it;
            }

            return acc;
        }

        template<class Container, class Function>
        auto sum(Function func, Container src) -> decltype(func(*begin(src)))
        {
            typedef decltype(func(*begin(src))) acc_type;
            typedef decltype(*begin(src)) con_type;
            auto it = begin(src);
            auto last = end(src);

            if (it == last)
                return acc_type();

            acc_type acc = func(*it);
            ++it;

            for (; it < last; ++it) {
                acc += func(*it);
            }

            return acc;
        }
    }

}

#endif // singleeyefitter_fun_h__
