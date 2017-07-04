

#ifndef singleeyefitter_CircleGoodness3D_h__
#define singleeyefitter_CircleGoodness3D_h__

#include "common/types.h"
#include "math/distance.h"
#include "common/constants.h"
#include "geometry/Sphere.h"
#include "fun.h"

#include <iostream>

namespace singleeyefitter {

    class CircleGoodness3D {
        typedef singleeyefitter::Sphere<double> Sphere;

        public:
            CircleGoodness3D( )
            {
            };

            double operator()(const Circle& circle, const Contours3D& contours) const
            {


                // Since contours don't represent extracted lines rather real contours around an "object", we actually weight each contour too much
                // Example: # Contour , - Line
                //         ###################
                //         #-----------------#
                //         ###################

                // If this would be a line segment of our Eye and the corresponding contour, the euclidian length of the contour doesn't represent the length of the line
                //  Althougth it looks like it's just double the length, it's not true for more complex lines.
                // Also if the line has branches, we can't tell how the contour is extracted.

                // Working in 2D we used the real raw edges too calculate the goodness, but for the 3D case we need something different.

                // Considering that the contours lie near the fitted circle border we wanna calculate the min and max opening angle of the contour arc.
                // For this we create a Sphere with Sphere center == circle center and Sphere radius = circle radius, further all contour points are
                // mapped to spherical coordinates.

                // Thus all contour points are represented with two angles and the radius. Where the interesting angle is the azimuth.
                // Further we can find all overlaps by comparing min and max of the azimuth angles and ignore doubled one.

                // This allows us to calculate the real coverage of the circumeference of the fitted circle, which tells us how much the contours support the current circle.


                // This matrix transforms the contour points from camera space to circle space
                // where the circle normal is alinged with the up vector , in our case the y is up
                Vector3 upVector(0,1,0);
                Eigen::Affine3d pointTransformation;
                pointTransformation = Eigen::Translation<double,3>( -circle.center );

                if( circle.normal != upVector){
                    Vector3 rotationAxis = circle.normal.cross(upVector).normalized();
                    double angle = std::acos(circle.normal.dot( upVector)); // angle in radians
                    pointTransformation =   Eigen::AngleAxisd( angle, rotationAxis ) * pointTransformation;

                }
                //visualize transformed circle and contours, remove const in function parameters
                // CircleGoodness get invalid if you uncomment this
                // To make it work again remove new_point transformation further down, otherwise it happens twice
                // circle.normal = pointTransformation.linear() * circle.normal;
                // circle.center = pointTransformation * circle.center;
                // for ( auto& contour : contours) {
                //     for(auto& point : contour){
                //         point = pointTransformation * point;
                //     }
                // }

                // Let's keep min and max of the azimuth (psi ) for every contour and ignore theta and r
                // theta and r are already contained in the residual of the plane fit (theta) and the circle variance (r)

                std::vector<std::pair<double,double>> contours_angles;
                const double circle_radius_squared = circle.radius*circle.radius;

                 for ( auto& contour : contours) {

                    double angle_min = std::numeric_limits<double>::infinity();
                    double angle_max = -1 ;
                    double angle_prev_point = -1;
                    for(const auto& point : contour){

                        const auto new_point = pointTransformation * point;
                        const double point_distance_squared = new_point.squaredNorm();

                        //skip point which are to far away form the circle boarder
                        // happens if a contour has a kink in it
                        // doesn't take into account if the contour get's nearer again
                        // but acctually shouldn't happen
                        const double ratio = point_distance_squared/circle_radius_squared;
                        const double ratio_factor = 0.075;
                        const double ratio_factor_max = 1 + ratio_factor;
                        const double ratio_factor_min = 1 - ratio_factor;

                         if( ratio > ratio_factor_max || ratio < ratio_factor_min )
                             continue;

                        //double theta = acos(point.y() / r);
                        //std::cout << "new point: "  << new_point << std::endl;

                        double psi = atan2(new_point.z(), new_point.x() );
                        if( psi < 0) psi +=  constants::TWO_PI; // range [0,2pi]
                        // std::cout << "psi: "  << psi << std::endl;
                        // std::cout << "angel max: "  << angle_max << std::endl;
                        // std::cout << "angel min: "  << angle_min << std::endl;
                        //std::cout << "prev angle : "  << angle_prev_point << std::endl;

                        // find wrap arounds ( contours going through 0 )
                        // if we find these, we split them
                        // we take some asumptions, one of them is that two consecutive points can't have a big angle, can't be the same or bigger than pi

                        if( angle_prev_point != -1 &&  angle_prev_point - psi  > constants::PI ){ // the line goes from 2pi to 0

                            // split here
                           contours_angles.push_back( {angle_min,  constants::TWO_PI} );
                           // start again at 0
                           angle_min = 0.0 ;
                           angle_max = psi;
                           // std::cout << "found positive wrap around" << std::endl;
                        }
                        if( angle_prev_point != -1 &&  angle_prev_point - psi   < -constants::PI ){ // the line goes from 0 to 2pi

                            // split here
                           contours_angles.push_back( {0.0, angle_max} );
                           // start again at 2pi
                           angle_min = psi ;
                           angle_max = constants::TWO_PI;
                           // std::cout << "found negative wrap around" << std::endl;
                        }

                        if( psi > angle_max) angle_max = psi;
                        if (psi < angle_min) angle_min = psi;

                        angle_prev_point = psi;
                    }
                    // std::cout << "add: " <<angle_min << " " << angle_max  << std::endl;
                    contours_angles.push_back( {angle_min,angle_max} );
                }


                // to eliminate overelapping contours look at each contour and keep track of overlaps
                // to not walk through n*n contours we sort them by increasing min angle
                std::sort(contours_angles.begin(), contours_angles.end(), [](const std::pair<double,double>& a, const std::pair<double,double>& b) { return a.first < b.first ;});


                double current_angle_min = std::numeric_limits<double>::infinity();
                double current_angle_max = -1 ;
                double angle_total  = 0;
                for( const auto&  angles : contours_angles ){
                    // std::cout << "current angle max " <<  current_angle_max << std::endl;
                    // std::cout << "current angle min " <<  current_angle_min << std::endl;
                    // std::cout << "angle first " <<  angles.first << std::endl;
                    // std::cout << "angle second " <<  angles.second << std::endl;
                    if( angles.first > current_angle_min && angles.second < current_angle_max ){ // contour is smaller ignore it
                       continue;
                    }

                    if( angles.first < current_angle_max && angles.second > current_angle_max ){ // there is a contour which overlapses and is longer
                        current_angle_max = angles.second;
                    }

                    if( angles.first > current_angle_max ){ // there is one starting outside of current contour

                        // calculate the angle of the previous contour and add it to the total

                        if( current_angle_max != -1 &&  current_angle_min != std::numeric_limits<double>::infinity() ){  // just add the previous one if there is one
                            double a = current_angle_max - current_angle_min;
                            // std::cout << "add amount " <<  a << std::endl;
                            angle_total += a;

                        }

                        current_angle_min = angles.first;
                        current_angle_max = angles.second;
                    }

                }
                // don't forget to add the last found angles
                if( current_angle_max != -1 &&  current_angle_min != std::numeric_limits<double>::infinity() ){
                    double a = current_angle_max - current_angle_min;
                    angle_total += a;
                }


                double goodness =  angle_total / constants::TWO_PI ;
                return goodness;
            }


            double operator()(const Circle& circle,  Edges3D& edges, double focalLength , Sphere sphere ) const
            {


                // We go the same way as in 2D and compare the amount of edges with circumference of the ellipse

                // project the circle back on the image plane
                Ellipse ellipse = Ellipse(project( circle, focalLength ));

                // filter edges
                // just use the edges on the plane created by the circle
                // and within a certain radius error
                //const double radiusTolerance = 0.1;
                //const Vector3 circleCenter = circle.center - sphere.center; // in coord system of the sphere
                const Vector3 planeNormal = circle.normal;
                const double planePointLength  = std::sqrt( sphere.radius * sphere.radius - circle.radius * circle.radius);
                const Vector3 planePoint = sphere.center +  planePointLength * planeNormal; // real circle center in the sphere

                auto circleFilter = [&]( const Vector3& e ){
                    Vector3 point = e - planePoint;
                    double length = point.norm();
                    Vector3 pointNormalized = point/ length;
                    double angleError = std::abs(pointNormalized.dot(planeNormal));
                    double radiusError = std::abs(1.0 - circle.radius / length );
                    return angleError <= 0.02 && radiusError <= 0.05;
                };

                edges = fun::filter( circleFilter , edges);

                const double circumference = ellipse.circumference();

                const double goodness  = edges.size() / circumference;
                return std::min(goodness, 1.0);
            }
    };

} // namespace singleeyefitter

#endif // singleeyefitter_CircleGoodness3D_h__
