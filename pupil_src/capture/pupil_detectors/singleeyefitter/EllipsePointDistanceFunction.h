

namespace singleeyefitter {


    template<typename Scalar>
    struct EllipsePointDistanceFunction {
        EllipsePointDistanceFunction(const Ellipse2D<Scalar>& el, Scalar x, Scalar y) : el(el), x(x), y(y) {}

        template <typename T>
        bool operator()(const T* const t, T* e) const
        {
            using std::sin;
            using std::cos;
            auto&& pt = pointAlongEllipse(el, t[0]);
            e[0] = norm(x - pt.x(), y - pt.y());
            return true;
        }

        const Ellipse2D<Scalar>& el;
        Scalar x, y;
    };
} // singleeyefitter
