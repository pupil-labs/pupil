#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <stdexcept>

namespace singleeyefitter {

class MakeString
{
public:
    std::stringstream stream;
    operator std::string() const { return stream.str(); }

    template<class T>
    MakeString& operator<<(T const& VAR) { stream << VAR; return *this; }
};

inline int pow2(int n)
{
    return 1 << n;
}

int random(int min, int max);
int random(int min, int max, unsigned int seed);

template<typename T>
std::vector<T> randomSubset(const std::vector<T>& src, typename std::vector<T>::size_type size)
{
    if (size > src.size())
        throw std::range_error("Subset size out of range");

    std::vector<T> ret;
    std::set<size_t> vals;

    for (size_t j = src.size() - size; j < src.size(); ++j)
    {
        size_t idx = random(0, j); // generate a random integer in range [0, j]

        if (vals.find(idx) != vals.end())
            idx = j;

        ret.push_back(src[idx]);
        vals.insert(idx);
    }

    return ret;
}

template<typename T>
std::vector<T> randomSubset(const std::vector<T>& src, typename std::vector<T>::size_type size, unsigned int seed)
{
    if (size > src.size())
        throw std::range_error("Subset size out of range");

    std::vector<T> ret;
    std::set<size_t> vals;

    for (size_t j = src.size() - size; j < src.size(); ++j)
    {
        size_t idx = random(0, j, seed+j); // generate a random integer in range [0, j]

        if (vals.find(idx) != vals.end())
            idx = j;

        ret.push_back(src[idx]);
        vals.insert(idx);
    }

    return ret;
}

} //namespace singleeyefitter

#endif // __UTILS_H__
