

#ifndef singleeyefitter_EllipseEvaluation2D_h__
#define singleeyefitter_EllipseEvaluation2D_h__

namespace singleeyefitter {

    class EllipseEvaluation2D {

        public:
            EllipseEvaluation2D(const cv::Rect& centerVarianz, const float roundness_ratio, const float size_min, const float size_max) :
                centerVarianz(centerVarianz), roundness_ratio(roundness_ratio), size_min(size_min), size_max(size_max)
            {
            };

            bool operator()(const cv::RotatedRect& ellipse) const
            {
                bool is_centered = centerVarianz.x < ellipse.center.x && ellipse.center.x < (centerVarianz.width + centerVarianz.x) &&
                                   centerVarianz.y < ellipse.center.y && ellipse.center.y < (centerVarianz.height + centerVarianz.y);

                if (is_centered) {
                    float max_radius = ellipse.size.height;
                    float min_radius = ellipse.size.width;
                    bool is_round = (min_radius / max_radius) >= roundness_ratio;

                    if (is_round) {
                        bool right_size = size_min <= max_radius && max_radius <= size_max;

                        if (right_size) return true;
                    }
                }

                return false;
            }

        private:
            cv::Rect centerVarianz;
            float roundness_ratio;
            float size_min, size_max;

    };

} // namespace singleeyefitter

#endif // singleeyefitter_EllipseEvaluation2D_h__
