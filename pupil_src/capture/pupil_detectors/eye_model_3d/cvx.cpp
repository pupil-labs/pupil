#include <singleeyefitter/cvx.h>
#include <singleeyefitter/math.h>

void singleeyefitter::cvx::getROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& roi, int borderType)
{
    cv::Rect bbSrc = boundingBox(src);
    cv::Rect validROI = roi & bbSrc;
    if (validROI == roi)
    {
        dst = cv::Mat(src, validROI);
    }
    else
    {
        // Figure out how much to add on for top, left, right and bottom
        cv::Point tl = roi.tl() - bbSrc.tl();
        cv::Point br = roi.br() - bbSrc.br();

        int top = std::max(-tl.y, 0);  // Top and left are negated because adding a border
        int left = std::max(-tl.x, 0); // goes "the wrong way"
        int right = std::max(br.x, 0);
        int bottom = std::max(br.y, 0);

        cv::Mat tmp(src, validROI);
        cv::copyMakeBorder(tmp, dst, top, bottom, left, right, borderType); 
        //setup.py can't find libopencv_higlui.so... problem here.
    }
}


float singleeyefitter::cvx::histKmeans(const cv::Mat_<float>& hist, int bin_min, int bin_max, int K, float init_centers[], cv::Mat_<uchar>& labels, cv::TermCriteria termCriteria)
{
    using namespace math;

    CV_Assert( hist.rows == 1 || hist.cols == 1 && K > 0 );

    labels = cv::Mat_<uchar>::zeros(hist.size());
    int nbins = hist.total();
    float binWidth = (bin_max - bin_min)/nbins;
    float binStart = bin_min + binWidth/2;

    cv::Mat_<float> centers(K, 1, init_centers, 4);

    int iters = 0;
    bool finalRun = false;
    while (true)
    {
        ++iters;
        cv::Mat_<float> old_centers = centers.clone();

        int i_bin;
        cv::Mat_<float>::const_iterator i_hist;
        cv::Mat_<uchar>::iterator i_labels;
        cv::Mat_<float>::iterator i_centers;
        uchar label;

        float sumDist = 0;
        int movedCount = 0;

        // Step 1. Assign each element a label
        for (i_bin = 0, i_labels = labels.begin(), i_hist = hist.begin();
             i_bin < nbins;
             ++i_bin, ++i_labels, ++i_hist)
        {
            float bin_val = binStart + i_bin*binWidth;
            float minDist = sq(bin_val - centers(*i_labels));
            int curLabel = *i_labels;

            for (label = 0; label < K; ++label)
            {
                float dist = sq(bin_val - centers(label));
                if (dist < minDist)
                {
                    minDist = dist;
                    *i_labels = label;
                }
            }

            if (*i_labels != curLabel)
                movedCount++;

            sumDist += (*i_hist) * std::sqrt(minDist);
        }

        if (finalRun)
            return sumDist;

        // Step 2. Recalculate centers
        cv::Mat_<float> counts(K, 1, 0.0f);
        for (i_bin = 0, i_labels = labels.begin(), i_hist = hist.begin();
             i_bin < nbins;
             ++i_bin, ++i_labels, ++i_hist)
        {
            float bin_val = binStart + i_bin*binWidth;

            centers(*i_labels) += (*i_hist) * bin_val;
            counts(*i_labels) += *i_hist;
        }
        for (label = 0; label < K; ++label)
        {
            if (counts(label) == 0)
                return std::numeric_limits<float>::infinity();

            centers(label) /= counts(label);
        }

        // Step 3. Detect termination criteria
        if (movedCount == 0)
            finalRun = true;
        else if (termCriteria.type | cv::TermCriteria::COUNT && iters >= termCriteria.maxCount)
            finalRun = true;
        else if (termCriteria.type | cv::TermCriteria::EPS)
        {
            float max_movement = 0;
            for (label = 0; label < K; ++label)
            {
                max_movement = std::max(max_movement, sq(centers(label) - old_centers(label)));
            }
            if (sqrt(max_movement) < termCriteria.epsilon)
                finalRun = true;
        }
    }
    return std::numeric_limits<float>::infinity();
}

cv::RotatedRect singleeyefitter::cvx::fitEllipse(const cv::Moments& m)
{
    using namespace math;

    cv::RotatedRect ret;

    ret.center.x = m.m10/m.m00;
    ret.center.y = m.m01/m.m00;

    double mu20 = m.m20/m.m00 - ret.center.x*ret.center.x;
    double mu02 = m.m02/m.m00 - ret.center.y*ret.center.y;
    double mu11 = m.m11/m.m00 - ret.center.x*ret.center.y;

    double common = std::sqrt(sq(mu20 - mu02) + 4*sq(mu11));

    ret.size.width = std::sqrt(2*(mu20 + mu02 + common));
    ret.size.height = std::sqrt(2*(mu20 + mu02 - common));

    double num, den;
    if (mu02 > mu20) {
        num = mu02 - mu20 + common;
        den = 2*mu11;
    }
    else {
        num = 2*mu11;
        den = mu20 - mu02 + common;
    }

    if (num == 0 && den == 0)
        ret.angle = 0;
    else
        ret.angle = (180/PI) * std::atan2(num,den);

    return ret;
}
cv::Vec2f singleeyefitter::cvx::majorAxis(const cv::RotatedRect& ellipse)
{
    return cv::Vec2f(ellipse.size.width*std::cos(PI/180*ellipse.angle), ellipse.size.width*std::sin(PI/180*ellipse.angle));
}
