#ifndef ELLIPSEGOODNESSFUNCTION_H__
#define ELLIPSEGOODNESSFUNCTION_H__


//#define DEBUG_ELLIPSE_GOODNESS
//#define USE_INLINED_ELLIPSE_DIST

#ifdef USE_INLINED_ELLIPSE_DIST
#define IF_INLINED_ELLIPSE_DIST(...) __VA_ARGS__
#else
#define IF_INLINED_ELLIPSE_DIST(...)
#endif


namespace singleeyefitter {


// Calculates the x crossings of a conic at a given y value. Returns the number of crossings (0, 1 or 2)
    template<typename Scalar>
    int getXCrossing(const Conic<Scalar>& conic, Scalar y, Scalar& x1, Scalar& x2)
    {
        using std::sqrt;
        Scalar a = conic.A;
        Scalar b = conic.B * y + conic.D;
        Scalar c = conic.C * y * y + conic.E * y + conic.F;
        Scalar det = b * b - 4 * a * c;

        if (det == 0) {
            x1 = -b / (2 * a);
            return 1;

        } else if (det < 0) {
            return 0;

        } else {
            Scalar sqrtdet = sqrt(det);
            x1 = (-b - sqrtdet) / (2 * a);
            x2 = (-b + sqrtdet) / (2 * a);
            return 2;
        }
    }


    namespace internal {
        template<class T> T ellipseGoodness(const Ellipse2D<T>& ellipse, const cv::Mat_<uint8_t>& eye, T band_width, T step_epsilon, scalar_tag);
        template<class T> T ellipseGoodness(const Ellipse2D<T>& ellipse, const cv::Mat_<uint8_t>& eye, typename ad_traits<T>::scalar band_width, typename ad_traits<T>::scalar step_epsilon, ceres_jet_tag);
    }

    namespace internal {
// Non autodiff version of ellipse goodness calculation
        template<class T>
        T ellipseGoodness(const Ellipse2D<T>& ellipse, const cv::Mat_<uint8_t>& eye, T band_width, T step_epsilon, scalar_tag)
        {
            using std::max;
            using std::min;
            using std::ceil;
            using std::floor;
            using std::sin;
            using std::cos;
            // Ellipses (and corresponding conics) delimiting the region in which the band masks will be non-zero
            Ellipse2D<T> outerEllipse = scaledMajorRadius(ellipse, ellipse.major_radius + ((band_width + step_epsilon) + 0.5));
            Ellipse2D<T> innerEllipse = scaledMajorRadius(ellipse, ellipse.major_radius - ((band_width + step_epsilon) + 0.5));
            Conic<T> outerConic(outerEllipse);
            Conic<T> innerConic(innerEllipse);
            // Variables for calculating the mean
            T sum_inner = T(0), count_inner = T(0), sum_outer = T(0), count_outer = T(0);
            // Only iterate over pixels within the outer ellipse's bounding box
            cv::Rect bb = bounding_box(outerEllipse);
            bb &= cv::Rect(-eye.cols / 2, -eye.rows / 2, eye.cols, eye.rows);
#ifndef USE_INLINED_ELLIPSE_DIST
            // Ellipse distance calculator
            EllipseDistCalculator<T> ellipDist(ellipse);
#else
            // Instead of calculating
            //     r * (1 - ||A(p - t)||)
            // we use
            //     (r - ||rAp - rAt||)
            // and precalculate r, rA and rAt.
            Eigen::Matrix<T, 2, 2> rA;
            T r = ellipse.major_radius;
            rA << r* cos(ellipse.angle) / ellipse.major_radius, r* sin(ellipse.angle) / ellipse.major_radius,
            -r* sin(ellipse.angle) / ellipse.minor_radius, r* cos(ellipse.angle) / ellipse.minor_radius;
            Eigen::Matrix<T, 2, 1> rAt = rA * ellipse.center;
            // Actually,
            ///    rAp - rAt = rA(0,y) + rA(x,0) - rAt
            // So, can perform a strength reduction to calculate rAp iteratively.
            // rA(0,y) - rAt, with y_0 = bb.y
            Eigen::Matrix<T, 2, 1> rA0yrAt(rA(0, 1) * bb.y - rAt[0], rA(1, 1) * bb.y - rAt[1]);
            // rA(1,0), for incrementing x
            Eigen::Matrix<T, 2, 1> rA10 = rA.col(0);
            // rA(0,1), for incrementing y
            Eigen::Matrix<T, 2, 1> rA01 = rA.col(1);
#endif

            for (int i = bb.y; i < bb.y + bb.height; ++i IF_INLINED_ELLIPSE_DIST(, rA0yrAt += rA01)) {
                // Image row pointer -- (0,0) is center of image, so shift accordingly
                const uint8_t* eye_i = eye[i + eye.rows / 2];
                // Only iterate over pixels between the inner and outer ellipse
                T ox1, ox2;
                int outerCrossings = getXCrossing<T>(outerConic, i, ox1, ox2);

                if (outerCrossings < 2) {
                    // If we don't cross the outer ellipse at all, exit early
                    continue;
                }

                T ix1, ix2;
                int innerCrossings = innerEllipse.minor_radius > 0 ? getXCrossing<T>(innerConic, i, ix1, ix2) : 0;
                // Define pairs of x values to iterate between
                std::vector<std::pair<int, int>> xpairs;

                if (innerCrossings < 2) {
                    // If we don't cross the inner ellipse, iterate between the two crossings of the outer ellipse
                    xpairs.emplace_back(max<int>(floor(ox1), bb.x), min<int>(ceil(ox2), bb.x + bb.width - 1));

                } else {
                    // Otherwise, iterate between outer-->inner, then inner-->outer.
                    xpairs.emplace_back(max<int>(floor(ox1), bb.x), min<int>(ceil(ix1), bb.x + bb.width - 1));
                    xpairs.emplace_back(max<int>(floor(ix2), bb.x), min<int>(ceil(ox2), bb.x + bb.width - 1));
                }

                // Go over x pairs (that is, outer-->outer or outer-->inner,inner-->outer)
                for (const auto& xpair : xpairs) {
                    // Pixel pointer, shifted accordingly
                    const uint8_t* eye_ij = eye_i + xpair.first + eye.cols / 2;
#ifdef USE_INLINED_ELLIPSE_DIST
                    // rA(0,y) + rA(x,0) - rAt, with x_0 = xpair.first
                    Eigen::Matrix<T, 2, 1> rApt(rA0yrAt(0) + rA(0, 0)*xpair.first, rA0yrAt(1) + rA(1, 0)*xpair.first);
#endif

                    for (int j = xpair.first; j <= xpair.second; ++j, ++eye_ij IF_INLINED_ELLIPSE_DIST(, rApt += rA10)) {
                        auto eye_ij_val = *eye_ij;

                        if (eye_ij_val > 200) {
                            // Ignore bright areas (i.e. glints)
                            continue;
                        }

#ifdef USE_INLINED_ELLIPSE_DIST
                        T dist = (r - norm(rApt(0), rApt(1)));
#else
                        T dist = ellipDist(T(j), T(i));
#endif
                        // Calculate mask values for each band
                        T Hellip = Heaviside(dist, step_epsilon);
                        T Houter = Heaviside(dist + band_width, step_epsilon);
                        T Hinner = Heaviside(dist - band_width, step_epsilon);
                        T outer_weight = (Houter - Hellip);
                        T inner_weight = (Hellip - Hinner);
                        sum_outer += outer_weight * eye_ij_val;
                        count_outer += outer_weight;
                        sum_inner += inner_weight * eye_ij_val;
                        count_inner += inner_weight;
                    }
                }
            }

            // Get mean values, defaulting to 255 and 0 if count_inner/count_outer are 0 (respectively)
            // Using 255 and 0 because these are the "worst" values, so some pixels will be preferred over none.
            T mu_inner = (count_inner == 0 ? 255 : sum_inner / count_inner);
            T mu_outer = (count_outer == 0 ? 0 : sum_outer / count_outer);

            // If count < 100 pixels, interpolate between mean value and "worst" value. This will push the
            // gradient away from small pixel counts in a vaguely smooth way.
            if (count_outer < 100) {
                mu_outer = math::lerp<T>(0, mu_outer, count_outer / 100.0);
            }

            if (count_inner < 100) {
                mu_inner = math::lerp<T>(255, mu_inner, count_inner / 100.0);
            }

            // Return difference of mean values
            return mu_outer - mu_inner;
        }

// Autodiff version of ellipse goodness calculation
        template<class Jet>
        Jet ellipseGoodness(const Ellipse2D<Jet>& ellipse, const cv::Mat_<uint8_t>& eye, typename ad_traits<Jet>::scalar band_width, typename ad_traits<Jet>::scalar step_epsilon, ceres_jet_tag)
        {
            using std::max;
            using std::min;
            using std::ceil;
            using std::floor;
#ifdef DEBUG_ELLIPSE_GOODNESS
            cv::Mat_<cv::Vec3b> eye_proc = cv::Mat_<cv::Vec3b>::zeros(eye.rows, eye.cols);
            cv::Mat_<cv::Vec3b> eye_H = cv::Mat_<cv::Vec3b>::zeros(eye.rows, eye.cols);
#endif
            typedef typename ad_traits<Jet>::scalar T;
            typedef Jet Jet_t;
            // A constant version of the ellipse
            Ellipse2D<T> constEllipse = toConst(ellipse);
            // Ellipses (and corresponding conics) delimiting the region in which the band masks will be non-zero
            Ellipse2D<T> constOuterEllipse = scaledMajorRadius(constEllipse, constEllipse.major_radius + ((band_width + step_epsilon) + 0.5));
            Ellipse2D<T> constInnerEllipse = scaledMajorRadius(constEllipse, constEllipse.major_radius - ((band_width + step_epsilon) + 0.5));
            Conic<T> constOuterConic(constOuterEllipse);
            Conic<T> constInnerConic(constInnerEllipse);
            // Variables for calculating the mean
            Jet_t sum_inner = Jet_t(0), count_inner = Jet_t(0), sum_outer = Jet_t(0), count_outer = Jet_t(0);
            // Only iterate over pixels within the outer ellipse's bounding box
            cv::Rect bb = bounding_box(constOuterEllipse);
            bb &= cv::Rect(-eye.cols / 2, -eye.rows / 2, eye.cols, eye.rows);
#ifndef USE_INLINED_ELLIPSE_DIST
            // Ellipse distance calculator
            EllipseDistCalculator<Jet_t> ellipDist(ellipse);
            EllipseDistCalculator<T> constEllipDist(constEllipse);
#else
            // Instead of calculating
            //     r * (1 - ||A(p - t)||)
            // we use
            //     (r - ||rAp - rAt||)
            // and precalculate r, rA and rAt.
            Eigen::Matrix<T, 2, 2> rA;
            T r = constEllipse.major_radius;
            rA << r* cos(constEllipse.angle) / constEllipse.major_radius, r* sin(constEllipse.angle) / constEllipse.major_radius,
            -r* sin(constEllipse.angle) / constEllipse.minor_radius, r* cos(constEllipse.angle) / constEllipse.minor_radius;
            Eigen::Matrix<T, 2, 1> rAt = rA * constEllipse.center;
            // And non-constant versions of the above
            Eigen::Matrix<Jet_t, 2, 2> rA_jet;
            Jet_t r_jet = ellipse.major_radius;
            rA_jet << r_jet* cos(ellipse.angle) / ellipse.major_radius, r_jet* sin(ellipse.angle) / ellipse.major_radius,
                   -r_jet* sin(ellipse.angle) / ellipse.minor_radius, r_jet* cos(ellipse.angle) / ellipse.minor_radius;
            Eigen::Matrix<Jet_t, 2, 1> rAt_jet = rA_jet * ellipse.center;
            // Actually,
            ///    rAp - rAt = rA(0,y) + rA(x,0) - rAt
            // So, can perform a strength reduction to calculate rAp iteratively.
            // rA(0,y) - rAt, with y_0 = bb.y
            Eigen::Matrix<T, 2, 1> rA0yrAt(rA(0, 1) * bb.y - rAt[0], rA(1, 1) * bb.y - rAt[1]);
            // rA(1,0), for incrementing x
            Eigen::Matrix<T, 2, 1> rA10 = rA.col(0);
            // rA(0,1), for incrementing y
            Eigen::Matrix<T, 2, 1> rA01 = rA.col(1);
#endif

            for (int i = bb.y; i < bb.y + bb.height; ++i IF_INLINED_ELLIPSE_DIST(, rA0yrAt += rA01)) {
                // Image row pointer -- (0,0) is center of image, so shift accordingly
                const uint8_t* eye_i = eye[i + eye.rows / 2];
                // Only iterate over pixels between the inner and outer ellipse
                T ox1, ox2;
                int outerCrossings = getXCrossing<T>(constOuterConic, i, ox1, ox2);

                if (outerCrossings < 2) {
                    // If we don't cross the outer ellipse at all, exit early
                    continue;
                }

                T ix1, ix2;
                int innerCrossings = constInnerEllipse.major_radius > 0 ? getXCrossing<T>(constInnerConic, i, ix1, ix2) : 0;
                // Define pairs of x values to iterate between
                std::vector<std::pair<int, int>> xpairs;

                if (innerCrossings < 2) {
                    // If we don't cross the inner ellipse, iterate between the two crossings of the outer ellipse
                    xpairs.emplace_back(max<int>(floor(ox1), bb.x), min<int>(ceil(ox2), bb.x + bb.width - 1));

                } else {
                    // Otherwise, iterate between outer-->inner, then inner-->outer.
                    xpairs.emplace_back(max<int>(floor(ox1), bb.x), min<int>(ceil(ix1), bb.x + bb.width - 1));
                    xpairs.emplace_back(max<int>(floor(ix2), bb.x), min<int>(ceil(ox2), bb.x + bb.width - 1));
                }

#ifdef USE_INLINED_ELLIPSE_DIST
                // Precalculate the gradient of
                //     rA(y,0) - rAt
                auto rAy0rAt_x_v = (rA_jet(0, 1).v * i - rAt_jet(0).v).eval();
                auto rAy0rAt_y_v = (rA_jet(1, 1).v * i - rAt_jet(1).v).eval();
#endif

                // Go over x pairs (that is, outer-->outer or outer-->inner,inner-->outer)
                for (const auto& xpair : xpairs) {
                    // Pixel pointer, shifted accordingly
                    const uint8_t* eye_ij = eye_i + xpair.first + eye.cols / 2;
#ifdef USE_INLINED_ELLIPSE_DIST
                    // rA(0,y) + rA(x,0) - rAt, with x_0 = xpair.first
                    Eigen::Matrix<T, 2, 1> rApt(rA0yrAt(0) + rA(0, 0)*xpair.first, rA0yrAt(1) + rA(1, 0)*xpair.first);
#endif

                    for (int j = xpair.first; j <= xpair.second; ++j, ++eye_ij IF_INLINED_ELLIPSE_DIST(, rApt += rA10)) {
                        T eye_ij_val = *eye_ij;

                        if (eye_ij_val > 200) {
                            // Ignore bright areas (i.e. glints)
                            continue;
                        }

                        // Calculate signed ellipse distance without gradient first, in case the gradient is 0
#ifdef USE_INLINED_ELLIPSE_DIST
                        T dist_const = (r - norm(rApt(0), rApt(1)));
#else
                        T dist_const = constEllipDist(T(j), T(i));
#endif

                        // Check if we are within step_epsilon of the edges of the bands. If yes, calculate
                        // the gradient. Otherwise, the gradient is known to be 0.
                        if (abs(dist_const) < step_epsilon
                                || abs(dist_const - band_width) < step_epsilon
                                || abs(dist_const + band_width) < step_epsilon) {
#ifdef USE_INLINED_ELLIPSE_DIST
                            // Calculate the gradients of rApt, and use those to get the dist
                            Jet_t rAxt_jet(rApt(0),
                                           rA_jet(0, 0).v * j + rAy0rAt_x_v);
                            Jet_t rAyt_jet(rApt(1),
                                           rA_jet(1, 0).v * j + rAy0rAt_y_v);
                            //Eigen::Matrix<Jet,2,1> rApt_jet2 = rA_jet*Eigen::Matrix<Jet,2,1>(Jet(j),Jet(i)) - rAt_jet;
                            Jet_t dist = (r_jet - norm(rAxt_jet, rAyt_jet));
                            //Jet_t dist2 = ellipDist(T(j), T(i));
#else
                            Jet_t dist = ellipDist(T(j), T(i));
#endif
                            // Calculate mask values and derivatives for each band
                            Jet_t Hellip = Heaviside(dist, step_epsilon);
                            Jet_t Houter = Heaviside(dist + band_width, step_epsilon);
                            Jet_t Hinner = Heaviside(dist - band_width, step_epsilon);
                            Jet_t outer_weight = (Houter - Hellip);
                            Jet_t inner_weight = (Hellip - Hinner);
                            // Inline the Jet operator+= to allow eigen expression and noalias magic.
                            sum_outer.a += outer_weight.a * eye_ij_val;
                            sum_outer.v.noalias() += outer_weight.v * eye_ij_val;
                            count_outer.a += outer_weight.a;
                            count_outer.v.noalias() += outer_weight.v;
                            sum_inner.a += inner_weight.a * eye_ij_val;
                            sum_inner.v.noalias() += inner_weight.v * eye_ij_val;
                            count_inner.a += inner_weight.a;
                            count_inner.v.noalias() += inner_weight.v;
#ifdef DEBUG_ELLIPSE_GOODNESS
                            eye_H(i + eye.rows / 2, j + eye.cols / 2)[2] = outer_weight.a * 255;
                            eye_H(i + eye.rows / 2, j + eye.cols / 2)[1] = inner_weight.a * 255;
                            eye_H(i + eye.rows / 2, j + eye.cols / 2)[0] = 255;
                            eye_proc(i + eye.rows / 2, j + eye.cols / 2)[2] = outer_weight.a * eye_ij_val;
                            eye_proc(i + eye.rows / 2, j + eye.cols / 2)[1] = inner_weight.a * eye_ij_val;
                            eye_proc(i + eye.rows / 2, j + eye.cols / 2)[0] = 255;
#endif

                        } else {
                            // Calculate mask values for each band
                            T Hellip = Heaviside(dist_const, step_epsilon);
                            T Houter = Heaviside(dist_const + band_width, step_epsilon);
                            T Hinner = Heaviside(dist_const - band_width, step_epsilon);
                            T outer_weight = (Houter - Hellip);
                            T inner_weight = (Hellip - Hinner);
                            sum_outer.a += outer_weight * eye_ij_val;
                            count_outer.a += outer_weight;
                            sum_inner.a += inner_weight * eye_ij_val;
                            count_inner.a += inner_weight;
#ifdef DEBUG_ELLIPSE_GOODNESS
                            eye_H(i + eye.rows / 2, j + eye.cols / 2)[2] = outer_weight * 255;
                            eye_H(i + eye.rows / 2, j + eye.cols / 2)[1] = inner_weight * 255;
                            eye_H(i + eye.rows / 2, j + eye.cols / 2)[0] = 0;
                            eye_proc(i + eye.rows / 2, j + eye.cols / 2)[2] = outer_weight * eye_ij_val;
                            eye_proc(i + eye.rows / 2, j + eye.cols / 2)[1] = inner_weight * eye_ij_val;
                            eye_proc(i + eye.rows / 2, j + eye.cols / 2)[0] = 255;
#endif
                        }
                    }
                }
            }

            // Get mean values, defaulting to 255 and 0 if count_inner/count_outer are 0 (respectively)
            // Using 255 and 0 because these are the "worst" values, so some pixels will be preferred over none.
            Jet mu_inner = (count_inner.a == 0 ? Jet(255) : sum_inner / count_inner);
            Jet mu_outer = (count_outer.a == 0 ? Jet(0) : sum_outer / count_outer);

            // If count < 100 pixels, interpolate between mean value and "worst" value. This will push the
            // gradient away from small pixel counts in a vaguely smooth way.
            if (count_outer.a < 100) {
                mu_outer = math::lerp<Jet>(Jet(0), mu_outer, count_outer / 100.0);
            }

            if (count_inner.a < 100) {
                mu_inner = math::lerp<Jet>(Jet(255), mu_inner, count_inner / 100.0);
            }

            // Return difference of mean values
            return mu_outer - mu_inner;
        }
    }

// Calculates the "goodness" of an ellipse.
//
// This is defined as the difference in region means:
//
//    μ⁻ - μ⁺
//
// where
//         Σ_p (H(d(p)+w) - H(d(p))) I(p)
//    μ⁻ = ------------------------------
//           Σ_p (H(d(p)+w) - H(d(p)))
//
//         Σ_p (H(d(p)+w) - H(d(p))) I(p)©
//    μ⁺ = ------------------------------
//           Σ_p (H(d(p)+w) - H(d(p)))
//
// (see eqs 16, 20, 21 in the PETMEI paper)
//
// The ellipse distance d(p) is defined as
//
//    d(p) = r * (1 - ||A(p - t)||)
//
// with r as the major radius and A as the matrix that transforms the ellipse to a unit circle.
//
//          ||A(p - t)||   maps the ellipse to a unit circle
//      1 - ||A(p - t)||   measures signed distance from unit circle edge
// r * (1 - ||A(p - t)||)  scales this to major radius of ellipse, for (roughly) pixel distance
//
    template<class T>
    inline T ellipseGoodness(const Ellipse2D<T>& ellipse, const cv::Mat_<uint8_t>& eye, typename ad_traits<T>::scalar band_width, typename ad_traits<T>::scalar step_epsilon)
    {
        // band_width     The width of each band (inner and outer)
        // step_epsilon   The epsilon of the soft step function
        return internal::ellipseGoodness<T>(ellipse, eye, band_width, step_epsilon, typename ad_traits<T>::ad_tag());
    }

    template<typename T>
    struct EllipseGoodnessFunction {
        T operator()(const Sphere<T>& eye, T theta, T psi, T pupil_radius, T focal_length, typename ad_traits<T>::scalar band_width, typename ad_traits<T>::scalar step_epsilon, const cv::Mat& mEye)
        {
            typedef Eigen::Matrix<T, 3, 1> Vector3;
            typedef typename ad_traits<T>::scalar Const;
            static const Vector3 camera_center(T(0), T(0), T(0));

            // Check for bounds. The worst possible value of ellipseGoodness is -255, so use that as a starting point for out-of-bounds pupils

            // Pupil radius must be positive
            if (pupil_radius <= Const(0)) {
                // Return -255 for radius == 0, and even lower values for
                // radius < 0
                // This should push the gradient towards positive radius,
                // rather than just returning flat -255
                return Const(-255.0) + pupil_radius;
            }

            Circle3D<T> pupil_circle = circleOnSphere(eye, theta, psi, pupil_radius);
            // Ellipse normal must point towards camera
            T normalDotPos = pupil_circle.normal.dot(camera_center - pupil_circle.center);

            if (normalDotPos <= Const(0)) {
                // Return -255 for normalDotPos == 0, and even lower values for
                // normalDotPos < 0
                // This should push the gradient towards positive normalDotPos,
                // rather than just returning flat -255
                return Const(-255.0) + normalDotPos;
            }

            // Angles should be in the range
            //    theta: 0 -> pi
            //      psi: -pi -> 0
            // If we're outside of this range AND radialDotEye > 0, then we must
            // have gone all the way around, so just return worst case (i.e as bad
            // as radialDotEye == -1) with additional penalty for how far out we
            // are, again to push the gradient back inwards.
            if (theta < Const(0) || theta > Const(constants::PI) || psi < Const(-constants::PI) || psi > Const(0)) {
                T ret = Const(-255.0) - (camera_center - pupil_circle.center).norm();

                if (theta < Const(0))
                    ret -= (Const(0) - theta);
                else if (theta > Const(constants::PI))
                    ret -= (theta - Const(constants::PI));

                if (psi < Const(-constants::PI))
                    ret -= (Const(-constants::PI) - psi);
                else if (psi > Const(0))
                    ret -= (psi - Const(0));
            }

            // Ok, everything looks good so far, calculate the actual goodness.
            Ellipse2D<T> pupil_ellipse(project(pupil_circle, focal_length));
            return ellipseGoodness<T>(pupil_ellipse, mEye, band_width, step_epsilon);
        }
    };


} // namespace singleeyefitter
#endif /* end of include guard: ELLIPSEGOODNESSFUNCTION_H__ */
