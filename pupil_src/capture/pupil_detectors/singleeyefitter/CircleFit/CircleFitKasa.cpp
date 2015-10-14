/* ---------------------------------------------------------------------------
* Revision          : 1.0
* Modul             : Least Square Circle Kasa C++
* Creation          : 10.01.2015
* Recent changes    : 10.01.2015
* ----------------------------------------------------------------------------
* LOG:
* ----------------------------------------------------------------------------
* Author            : Dennis Luensch
* Contact           : dennis.luensch@gmail.com
* ----------------------------------------------------------------------------
* Tabsize           : 4
* Charset           : Windows
* ------------------------------------------------------------------------- */

#include <iostream>
#include <math.h>
#include <vector>
#include "common/types.h"
#include <Eigen/Dense>

using namespace std;

/**
 * Fit a circle in a set of points. You need a minimum of 1 point to fit a circle.
 *
 * @param points Is the set of points.
 * @param midpoint Is the fitted midpoint of the circle.
 * @param Returns true, if no error occur. An error occurs, if the points vector is empty.
 */
bool solveLeastSquaresCircleKasa(const  std::vector<singleeyefitter::Vector2> &points, singleeyefitter::Vector2 &midpoint, double &radius)
{
    int length = points.size();
    double x1;
    double x2;
    double x3;
    Eigen::MatrixXd AFill(3, length);
    Eigen::MatrixXd A(length, 3);
    Eigen::VectorXd AFirst(length);
    Eigen::VectorXd ASec(length);
    Eigen::VectorXd AFirstSquared(length);
    Eigen::VectorXd ASecSquared(length);
    Eigen::VectorXd ASquaredRes(length);
    Eigen::VectorXd b(length);
    Eigen::VectorXd c(3);
    bool ok = true;

    if (length > 1)
    {
        for (int i = 0; i < length; i++)
        {
            AFill(0, i) = points[i](0);
            AFill(1, i) = points[i](1);
            AFill(2, i) = 1;
        }

        A = AFill.transpose();

        for (int i = 0; i < length; i++)
        {
            AFirst(i) = A(i, 0);
            ASec(i) = A(i, 1);
        }

        for (int i = 0; i < length; i++)
        {
            AFirstSquared(i) = AFirst(i) * AFirst(i);
            ASecSquared(i) = ASec(i) * ASec(i);
        }

        b = AFirstSquared + ASecSquared;

        c = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        x1 = c(0);
        midpoint(0) = x1 * 0.5;
        x2 = c(1);
        midpoint(1) = x2 * 0.5;
        x3 = c(2);
        radius = sqrt((x1 * x1 + x2 * x2) / 4 + x3);
    }
    else
    {
        ok = false;
    }

    return ok;
}

int main()
{
    std::vector<singleeyefitter::Vector2> samplePoints;
    singleeyefitter::Vector2 midpoint;
    double radius;
    bool ok;

    samplePoints.push_back(singleeyefitter::Vector2(6.5, 6));
    samplePoints.push_back(singleeyefitter::Vector2(6, 3));
    samplePoints.push_back(singleeyefitter::Vector2(3.5, 2));
    samplePoints.push_back(singleeyefitter::Vector2(0.5, 3));
    samplePoints.push_back(singleeyefitter::Vector2(1, 5.5));
    samplePoints.push_back(singleeyefitter::Vector2(3, 7.5));

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    ok = solveLeastSquaresCircleKasa(samplePoints, midpoint, radius);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "time: " << elapsed_seconds.count() << std::endl;
    if (ok)
    {
        cout << "x: " << midpoint(0) << "  |  y: " << midpoint(1) << "  |  radius: " << radius << std::endl;
    }

    getchar();

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
