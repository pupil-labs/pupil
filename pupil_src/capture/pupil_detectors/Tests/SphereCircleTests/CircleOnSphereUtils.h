


std::vector<singleeyefitter::Vector3> createCirclePointsOnSphere( singleeyefitter::Vector2 center, double opening_angle_alpha,  int amount ){

    using namespace singleeyefitter;
    using std::sin;
    using std::cos;

    // http://math.stackexchange.com/questions/643130/circle-on-sphere
    double alpha = opening_angle_alpha;
    double beta = center[0];
    double gamma = center[1];

    double sin_a = sin(alpha);
    double cos_a = cos(alpha);
    double sin_b = sin(beta);
    double cos_b = cos(beta);
    double sin_g = sin(gamma);
    double cos_g = cos(gamma);

    std::vector<Vector3> points;

    for (int i = 0; i < amount; ++i)
    {

        double t = 2.0 * M_PI * i/amount;
        double cos_t = cos(t);
        double sin_t = sin(t);
        double x = sin_a * cos_b * cos_g * cos_t + sin_a * sin_g * sin_t - cos_a * sin_b * cos_g;
        double y = - sin_a * cos_b * sin_g * cos_t + sin_a * cos_g * sin_t + cos_a * sin_b * sin_g;
        double z = sin_a * sin_b * cos_t + cos_a * cos_b;

        points.emplace_back(x,y,z);
    }

    return points;

}
