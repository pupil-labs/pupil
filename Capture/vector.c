struct Vector
{
    double x, y;
};


double getRotateAngle(Vector vec1, Vector vec2)
{
const double epsilon = 1.0e-6;
const double PI = acos(-1.0); // 180 degree
double angle = 0;

// normalize
Vector norVec1, norVec2;
norVec1.x = vec1.x / sqrt(pow(vec1.x, 2) + pow(vec1.y, 2));
norVec1.y = vec1.y / sqrt(pow(vec1.x, 2) + pow(vec1.y, 2));
norVec2.x = vec2.x / sqrt(pow(vec2.x, 2) + pow(vec2.y, 2));
norVec2.y = vec2.y / sqrt(pow(vec2.x, 2) + pow(vec2.y, 2));

// dot product
double dotProd = (norVec1.x * norVec2.x) + (norVec1.y * norVec2.y);
if ( abs(dotProd - 1.0) <= epsilon )
    angle = 0;
else if ( abs(dotProd + 1.0) <= epsilon )
    angle = PI;
else {
    double cross = 0;
    angle = acos(dotProd);
    //cross product (clockwise or counter-clockwise)
    cross = (norVec1.x * norVec2.y) - (norVec2.x * norVec1.y);

    if (cross < 0) // vec1 rotate clockwise to vec2
            angle = 2 * PI - angle;
    }

    return angle*(180 / PI);

}