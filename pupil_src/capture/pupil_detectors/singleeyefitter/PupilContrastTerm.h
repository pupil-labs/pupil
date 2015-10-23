#ifndef PUPILCONTRASTTERM_H__
#define PUPILCONTRASTTERM_H__


#include "EllipseGoodnessFunction.h"
#include "Geometry/Sphere.h"


#include <spii/spii.h>
#include <spii/term.h>
#include <spii/function.h>
#include <spii/solver.h>

namespace singleeyefitter {

    template<bool has_eye_var = true>
    struct PupilContrastTerm : public spii::Term {
        const Sphere<double>& init_eye;
        double focal_length;
        const cv::Mat eye_image;
        double band_width;
        double step_epsilon;

        int eye_var_idx() const { return has_eye_var ? 0 : -1; }
        int pupil_var_idx() const { return has_eye_var ? 1 : 0; }

        PupilContrastTerm(const Sphere<double>& eye, double focal_length, cv::Mat eye_image, double band_width, double step_epsilon) :
            init_eye(eye),
            focal_length(focal_length),
            eye_image(eye_image),
            band_width(band_width),
            step_epsilon(step_epsilon)
        {}

        virtual int number_of_variables() const override
        {
            int nvars = 1; // This pupil params

            if (has_eye_var)
                nvars++; // Eye params

            return nvars;
        }
        virtual int variable_dimension(int var) const override
        {
            if (var == eye_var_idx()) // Eye params (x,y,z)
                return 3;

            if (var == pupil_var_idx()) // This pupil params (theta, psi, r)
                return 3;

            return -1;
        };
        virtual double evaluate(double* const* const vars) const override
        {
            auto& pupil_vars = vars[pupil_var_idx()];
            auto eye = init_eye;

            if (has_eye_var) {
                auto& eye_vars = vars[eye_var_idx()];
                eye.center = Sphere<double>::Vector(eye_vars[0], eye_vars[1], eye_vars[2]);
            }

            EllipseGoodnessFunction<double> goodnessFunction;
            auto theta = pupil_vars[0];
            auto psi = pupil_vars[1];
            auto r = pupil_vars[2];
            auto goodness = goodnessFunction(eye,
                                             theta, psi, r,
                                             focal_length,
                                             band_width, step_epsilon,
                                             eye_image);
            return -goodness;
        }
        virtual double evaluate(double* const* const vars, std::vector<Eigen::VectorXd>* gradient) const override
        {
            auto& pupil_vars = vars[pupil_var_idx()];
            double contrast_goodness_a;
            Eigen::Matrix<double, 3, 1> eye_contrast_goodness_v;
            Eigen::Matrix<double, 3, 1> pupil_contrast_goodness_v;

            // Get region contrast goodness using EllipseGoodnessFunction.
            if (has_eye_var) {
                // If varying the eye parameters, calculate the gradient wrt. to 6 params (3 eye + 3 pupil)
                typedef ceres::Jet<double, 6> EyePupilJet;
                auto& eye_vars = vars[eye_var_idx()];
                Eigen::Matrix<EyePupilJet, 3, 1> eye_pos(EyePupilJet(eye_vars[0], 0), EyePupilJet(eye_vars[1], 1), EyePupilJet(eye_vars[2], 2));
                Sphere<EyePupilJet> eye(eye_pos, EyePupilJet(init_eye.radius));
                EyePupilJet contrast_goodness;
                {
                    EllipseGoodnessFunction<EyePupilJet> goodnessFunction;
                    auto theta = EyePupilJet(pupil_vars[0], 3);
                    auto psi = EyePupilJet(pupil_vars[1], 4);
                    auto r = EyePupilJet(pupil_vars[2], 5);
                    contrast_goodness = goodnessFunction(eye,
                                                         theta, psi, r,
                                                         EyePupilJet(focal_length),
                                                         band_width, step_epsilon,
                                                         eye_image);
                }
                contrast_goodness_a = contrast_goodness.a;
                eye_contrast_goodness_v = contrast_goodness.v.segment<3>(0);
                pupil_contrast_goodness_v = contrast_goodness.v.segment<3>(3);

            } else {
                // Otherwise, calculate the gradient wrt. to the 3 pupil params
                typedef ::ceres::Jet<double, 3> PupilJet;
                Eigen::Matrix<PupilJet, 3, 1> eye_pos(PupilJet(init_eye.center[0]), PupilJet(init_eye.center[1]), PupilJet(init_eye.center[2]));
                Sphere<PupilJet> eye(eye_pos, PupilJet(init_eye.radius));
                PupilJet contrast_goodness;
                {
                    EllipseGoodnessFunction<PupilJet> goodnessFunction;
                    auto theta = PupilJet(pupil_vars[0], 0);
                    auto psi = PupilJet(pupil_vars[1], 1);
                    auto r = PupilJet(pupil_vars[2], 2);
                    contrast_goodness = goodnessFunction(eye,
                                                         theta, psi, r,
                                                         PupilJet(focal_length),
                                                         band_width, step_epsilon,
                                                         eye_image);
                }
                contrast_goodness_a = contrast_goodness.a;
                pupil_contrast_goodness_v = contrast_goodness.v;
            }

            double goodness;
            auto& eye_gradient = (*gradient)[eye_var_idx()];
            auto& pupil_gradient = (*gradient)[pupil_var_idx()];
            // No smoothness term, goodness and gradient are based only on frame goodness
            goodness = contrast_goodness_a;

            if (has_eye_var)
                eye_gradient = eye_contrast_goodness_v;

            pupil_gradient = pupil_contrast_goodness_v;
            // Flip sign to change goodness into cost (i.e. maximising into minimising)
            auto cost = -goodness;

            for (int i = 0; i < number_of_variables(); ++i) {
                (*gradient)[i] = -(*gradient)[i];
            }

            return cost;
        }
        virtual double evaluate(double* const* const variables,
                                std::vector<Eigen::VectorXd>* gradient,
                                std::vector< std::vector<Eigen::MatrixXd> >* hessian) const override
        {
            throw std::runtime_error("Not implemented");
        }

    };

} // singleeyefitter
#endif /* end of include guard: PUPILCONTRASTTERM_H__ */
