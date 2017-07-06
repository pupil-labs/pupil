/*!
________________________________________________________________________________

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
________________________________________________________________________________

The morphological skeleton of an image is the set of its non-zero pixels
which are equidistant to its boundaries.
More info: http://en.wikipedia.org/wiki/Topological_skeleton

Thinning an image consits in reducing its non-zero pixels to their
morphological skeleton.
More info: http://en.wikipedia.org/wiki/Thinning_(morphology)

This class is derived from https://github.com/arnaud-ramey/voronoi,
and contains the fast Guo Hall thinner algorithm

 */

#ifndef GUOHALLTHINNER_H__
#define GUOHALLTHINNER_H__


#include <deque>
#include <opencv2/core.hpp>
#include "ImageContour.h"
#include "cvx.h"

namespace singleeyefitter {

    class GuoHallThinner {
        public:
            //! a constant not limiting the number of iterations of an implementation
            static const int NOLIMIT = INT_MAX;

            //! default construtor
            GuoHallThinner()
            {
                _has_converged = false;
            }

            bool thin(const cv::Mat1b& img,
                      bool crop_img_before = true,
                      int max_iters = NOLIMIT)
            {

                cv::Mat subImage = img;

                if (crop_img_before) {

                    cv::Rect bbox  = cv::Rect(0, 0, img.cols, img.rows); // init if we find none
                    cvx::getRoiWithoutBorder(img, bbox);

                    cv::Mat croped_img;
                    // skeleton contour ignores a border of one pixel, so let's try to add one
                    // skelcontour needs a continues image, so copy it.
                    subImage = cv::Mat(img, bbox).adjustROI(1, 1, 1, 1);
                    subImage.copyTo(croped_img);

                    skelcontour.from_image_C4(croped_img);

                } else {
                    skelcontour.from_image_C4(img);

                }

                int cols = skelcontour.cols, rows = skelcontour.rows;
                // clear queues
                uchar* skelcontour_data = skelcontour.data;

                int niters = 0;
                bool change_made = true;

                while (change_made && niters < max_iters) {
                    change_made = false;

                    for (unsigned short iter = 0; iter < 2; ++iter) {
                        uchar* skelcontour_ptr = skelcontour_data;
                        rows_to_set.clear();
                        cols_to_set.clear();

                        // for each point in skelcontour, check if it needs to be changed
                        for (int row = 1; row < rows; ++row) {
                            for (int col = 1; col < cols; ++col) {
                                if (*skelcontour_ptr++ == ImageContour::CONTOUR &&
                                        need_set_guo_hall(skelcontour_data, iter, col, row, cols)) {
                                    cols_to_set.push_back(col);
                                    rows_to_set.push_back(row);
                                }
                            }
                        }

                        // set all points in rows_to_set (of skel)
                        unsigned int rows_to_set_size = rows_to_set.size();

                        for (unsigned int pt_idx = 0; pt_idx < rows_to_set_size; ++pt_idx) {
                            if (!change_made)
                                change_made = (skelcontour(rows_to_set[pt_idx], cols_to_set[pt_idx]));

                            skelcontour.set_point_empty_C4(rows_to_set[pt_idx], cols_to_set[pt_idx]);
                        }



                        if ((niters++) >= max_iters) // must be at the end of the loop
                            break;
                    }
                }

                // copy the skeleton back to the original image region
                skelcontour.copyTo(subImage);
                _has_converged = !change_made;
                return true;
            }



            /*! \return true if last thin() stopped because the algo converged,
             * and not because of the max_iters param.
             */
            inline bool hasConverged() const { return _has_converged; }

        protected:

            bool /*inline*/ need_set_guo_hall(uchar*  skeldata, int iter, int col, int row, int cols)
            {
                //uchar
                bool
                p2 = skeldata[(row - 1) * cols + col],
                p3 = skeldata[(row - 1) * cols + col + 1],
                p4 = skeldata[row     * cols + col + 1],
                p5 = skeldata[(row + 1) * cols + col + 1],
                p6 = skeldata[(row + 1) * cols + col],
                p7 = skeldata[(row + 1) * cols + col - 1],
                p8 = skeldata[row     * cols + col - 1],
                p9 = skeldata[(row - 1) * cols + col - 1];

                int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                         (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
                int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
                int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
                int N  = N1 < N2 ? N1 : N2;
                int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

                return (C == 1 && (N >= 2 && N <= 3) && m == 0);
            }

            bool _has_converged;
            ImageContour skelcontour;
            //! list of keys to set to 0 at the end of the iteration
            std::deque<int> cols_to_set;
            std::deque<int> rows_to_set;
    };

} // singleeyefitter

#endif /* end of include guard: GUOHALLTHINNER_H__ */
