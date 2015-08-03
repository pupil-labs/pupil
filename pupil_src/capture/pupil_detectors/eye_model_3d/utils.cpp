#include <singleeyefitter/utils.h>

#include <random>

static std::mt19937 static_gen;
int singleeyefitter::random(int min, int max)
{
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(static_gen);
}
int singleeyefitter::random(int min, int max, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(gen);
}
