
#ifndef PUPILANTHROTERM_H__
#define PUPILANTHROTERM_H__


#include <spii/spii.h>
#include <spii/term.h>
#include <spii/function.h>
#include <spii/solver.h>


namespace singleeyefitter {

    // Anthropomorphic term
    struct PupilAnthroTerm : public spii::Term {
        double mean;
        double sigma;
        double scale;

        PupilAnthroTerm(double mean, double sigma, double scale) : mean(mean), sigma(sigma), scale(scale)
        {}

        virtual int number_of_variables() const override
        {
            int nvars = 1; // This pupil params
            return nvars;
        }
        virtual int variable_dimension(int var) const override
        {
            if (var == 0) // This pupil params (r)
                return 3;

            return -1;
        }
        virtual double evaluate(double* const* const vars) const override
        {
            using math::sq;
            auto r = vars[0][2];
            auto radius_anthro_goodness = exp(-sq(r - mean) / sq(sigma));
            double goodness = radius_anthro_goodness;
            // Flip sign to change goodness into cost (i.e. maximising into minimising)
            auto cost = -goodness * scale;
            return cost;
        }
        virtual double evaluate(double* const* const vars, std::vector<Eigen::VectorXd>* gradient) const override
        {
            using math::sq;
            auto r = ceres::Jet<double, 1>(vars[0][2], 0);
            auto radius_anthro_goodness = exp(-sq(r - mean) / sq(sigma));
            double goodness = radius_anthro_goodness.a;
            (*gradient)[0].segment<1>(2) = radius_anthro_goodness.v;
            // Flip sign to change goodness into cost (i.e. maximising into minimising)
            auto cost = -goodness * scale;

            for (int i = 0; i < number_of_variables(); ++i) {
                (*gradient)[i] = -(*gradient)[i] * scale;
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

#endif /* end of include guard: PUPILANTHROTERM_H__ */
