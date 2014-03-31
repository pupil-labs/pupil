/*
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
*/

/*
 *  V4L2 video capture example
 *
 *  This program can be used and distributed without restrictions.
 *
 *      This program is provided with the V4L2 API
 * see http://linuxtv.org/docs.php for more information
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <time.h>

#include <linux/videodev2.h>
#include <libv4l2.h>
#define CLEAR(x) memset(&(x), 0, sizeof(x))

#ifndef V4L2_PIX_FMT_H264
#define V4L2_PIX_FMT_H264     v4l2_fourcc('H', '2', '6', '4') /* H264 with start codes */
#endif


struct buffer {
	void   *start;
	size_t length;
};

// static int fd = -1;
static char            *dev_name;
struct buffer          *buffers;
static unsigned int n_buffers;

static void errno_exit(const char *s)
{
	fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
	exit(EXIT_FAILURE);
}

int xioctl(int fh, int request, void *arg)
{
	int r;

	do {
		r = v4l2_ioctl(fh, request, arg);
	} while (-1 == r && EINTR == errno);

	return r;
}


void *get_buffer(int fd,struct v4l2_buffer *buf){
	for (;; ) {
		fd_set fds;
		struct timeval tv;
		int r;

		FD_ZERO(&fds);
		FD_SET(fd, &fds);

		/* Timeout. */
		tv.tv_sec = 2;
		tv.tv_usec = 0;

		r = select(fd + 1, &fds, NULL, NULL, &tv);

		if (-1 == r) {
			if (EINTR == errno)
				continue;
			return 0;
			// errno_exit("select");
		}

		if (0 == r) {
			return 0;
			// fprintf(stderr, "select timeout\n");
			// exit(EXIT_FAILURE);
		}
		CLEAR(*buf);

		buf->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf->memory = V4L2_MEMORY_MMAP;

		if (-1 == xioctl(fd, VIDIOC_DQBUF, buf)) {
			switch (errno) {
			case EAGAIN: // no buffer available yet.
				return 0;

			case EIO:
			/* Could ignore EIO, see spec. */

			/* fall through */

			default:
				return 0;
				// errno_exit("VIDIOC_DQBUF");
			}
		}
		// printf("flags %u \n", buf->flags);
		assert(buf->index < n_buffers);
		// printf("time %ld %ld %u \n",
		   // buf->timestamp.tv_sec,buf->timestamp.tv_usec,buf->index);
		// process_image(buffers[buf.index].start, buf.bytesused);

		struct timespec raw_tv;
		clock_gettime(CLOCK_MONOTONIC, &raw_tv);

		// CLOCK_REALTIME
		// - can jump
		// - can slew
		// - if ntp is running this clock is always kept close to GMT. even if hardware is not 100% correct, ntp will correct everything over time.

		// CLOCK_MONOTONIC
		// - cannot jump
		// - can slew !!! (because of ntp)
		// - it is not kept in sync with GMT. but the "speed" of seconds is kept in sync with GMT by varying it constantly by ntp.

		// CLOCK_MONOTONIC_RAW
		// - cannot jump
		// - cannot slew !
		// - the speed of seconds is not the same as the speed of GMT seconds since the hardware timer is never 100% exact and ntp daemon does NOT have influence here


		// printf("current time %ld, %ld\n", raw_tv.tv_sec, raw_tv.tv_nsec);
		// buf->timestamp.tv_sec = (long) raw_tv.tv_sec;
		// buf->timestamp.tv_usec = raw_tv.tv_nsec/1000;
		return buffers[buf->index].start;
	}
}


int release_buffer(int fd, struct v4l2_buffer *buf){
	if (-1 == xioctl(fd, VIDIOC_QBUF, buf))
			// errno_exit("VIDIOC_QBUF");
			return 0;
	return 1;
}


int stop_capturing(int fd)
{
	enum v4l2_buf_type type;
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
		// errno_exit("VIDIOC_STREAMOFF");
		return 0;
	return 1;
}

int start_capturing(int fd)
{
	unsigned int i;
	enum v4l2_buf_type type;
	for (i = 0; i < n_buffers; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
			// errno_exit("VIDIOC_QBUF");
			return -1;
	}
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
		errno_exit("VIDIOC_STREAMON");
	return 1;
}

void uninit_device(int vd)
{
	unsigned int i;
	for (i = 0; i < n_buffers; ++i)
		if (-1 == v4l2_munmap(buffers[i].start, buffers[i].length))
			// errno_exit("munmap");
	free(buffers);
}


void init_mmap(int fd)
{
	struct v4l2_requestbuffers req;

	CLEAR(req);

	req.count = 4;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
		if (EINVAL == errno) {
			fprintf(stderr, "%s does not support "
			        "memory mapping\n", dev_name);
			exit(EXIT_FAILURE);
		} else {
			errno_exit("VIDIOC_REQBUFS");
		}
	}

	if (req.count < 2) {
		fprintf(stderr, "Insufficient buffer memory on %s\n",
		        dev_name);
		exit(EXIT_FAILURE);
	}

	buffers = calloc(req.count, sizeof(*buffers));

	if (!buffers) {
		fprintf(stderr, "Out of memory\n");
		exit(EXIT_FAILURE);
	}

	for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
		struct v4l2_buffer buf;

		CLEAR(buf);

		buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory      = V4L2_MEMORY_MMAP;
		buf.index       = n_buffers;

		if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
			errno_exit("VIDIOC_QUERYBUF");

		buffers[n_buffers].length = buf.length;
		buffers[n_buffers].start =
		        v4l2_mmap(NULL /* start anywhere */,
		             buf.length,
		             PROT_READ | PROT_WRITE /* required */,
		             MAP_SHARED /* recommended */,
		             fd, buf.m.offset);

		if (MAP_FAILED == buffers[n_buffers].start)
			errno_exit("mmap");
	}
}



int verify_device(int fd)
{
	struct v4l2_capability cap;
	// struct v4l2_cropcap cropcap;
	// struct v4l2_crop crop;
	// unsigned int min;

	if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
		if (EINVAL == errno) {
			fprintf(stderr, "%s is no V4L2 device\n",
			        dev_name);
			exit(EXIT_FAILURE);
		} else {
			errno_exit("VIDIOC_QUERYCAP");
		}
	}

	if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
		fprintf(stderr, "%s is no video capture device\n",
		        dev_name);
		exit(EXIT_FAILURE);
	}


	if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
		fprintf(stderr, "%s does not support streaming i/o\n",
		        dev_name);
		exit(EXIT_FAILURE);
	}


	return 0;
}

double get_time_monotonic(void){
	struct timespec raw_tv;
	clock_gettime(CLOCK_MONOTONIC, &raw_tv);
	return raw_tv.tv_sec + raw_tv.tv_nsec * 1e-9;
}

// void init_device(int fd,struct v4l2_format *fmt, struct v4l2_streamparm *params)
// {
// 	struct v4l2_capability cap;
// 	struct v4l2_cropcap cropcap;
// 	struct v4l2_crop crop;
// 	unsigned int min;

// 	if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
// 		if (EINVAL == errno) {
// 			fprintf(stderr, "%s is no V4L2 device\n",
// 			        dev_name);
// 			exit(EXIT_FAILURE);
// 		} else {
// 			errno_exit("VIDIOC_QUERYCAP");
// 		}
// 	}

// 	if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
// 		fprintf(stderr, "%s is no video capture device\n",
// 		        dev_name);
// 		exit(EXIT_FAILURE);
// 	}


// 	if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
// 		fprintf(stderr, "%s does not support streaming i/o\n",
// 		        dev_name);
// 		exit(EXIT_FAILURE);
// 	}



// 	/* Select video input, video standard and tune here. */


// 	CLEAR(cropcap);

// 	cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

// 	if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
// 		crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
// 		crop.c = cropcap.defrect; /* reset to default */

// 		if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop)) {
// 			switch (errno) {
// 			case EINVAL:
// 				/* Cropping not supported. */
// 				break;
// 			default:
// 				/* Errors ignored. */
// 				break;
// 			}
// 		}
// 	} else {
// 		/* Errors ignored. */
// 	}

// 	struct v4l2_format fmt;
// 	CLEAR(*fmt);
// 	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

// 	fmt.fmt.pix.width       = *width; //replace
// 	fmt.fmt.pix.height      = *height; //replace
// 	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24; //replace
// 	fmt.fmt.pix.field       = V4L2_FIELD_ANY;

// 	if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
// 		errno_exit("VIDIOC_S_FMT");


// 	if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt))
// 		errno_exit("VIDIOC_G_FMT");
// 	fprintf(stderr, "Set resolution: %u x %u \r\n",fmt.fmt.pix.width, fmt.fmt.pix.height );
// 	*width = fmt.fmt.pix.width;
// 	*height = fmt.fmt.pix.height;

// 	/* Buggy driver paranoia. */
// 	min = fmt.fmt.pix.width * 2;
// 	if (fmt.fmt.pix.bytesperline < min)
// 		fmt.fmt.pix.bytesperline = min;
// 	min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
// 	if (fmt.fmt.pix.sizeimage < min)
// 		fmt.fmt.pix.sizeimage = min;


// 	struct v4l2_streamparm params;
// 	CLEAR(params);
// 	params.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

// 	if (xioctl(fd, VIDIOC_G_PARM, &params) == -1)
// 		fprintf(stderr, "Unable to get stream parameters for device: %s", dev_name);

// 	params.parm.capture.timeperframe.numerator = *fps_numer;
// 	params.parm.capture.timeperframe.denominator = *fps_denom;

// 	if (xioctl(fd, VIDIOC_S_PARM, &params) == -1)
// 		fprintf(stderr, "Unable to set stream parameters for device: %s", dev_name);

// 	if (xioctl(fd, VIDIOC_G_PARM, &params) == -1)
// 		fprintf(stderr, "Unable to set stream parameters for device: %s", dev_name);
// 	fprintf(stderr, "framerate: %i/%i \n",params.parm.capture.timeperframe.numerator ,params.parm.capture.timeperframe.denominator  );
// 	*fps_numer =params.parm.capture.timeperframe.numerator;
// 	*fps_denom = params.parm.capture.timeperframe.denominator;
// 	init_mmap(fd);
// }

int close_device(int fd)
{
	if (-1 == close(fd))
		return 0;
		// errno_exit("close");

	fd = -1;
	return fd;
}

int open_device(char *dev_name_)
{
	struct stat st;
	int fd = -1;
	dev_name = dev_name_;
	if (-1 == stat(dev_name, &st)) {
		fprintf(stderr, "Cannot identify '%s': %d, %s\n",
		        dev_name, errno, strerror(errno));
		exit(EXIT_FAILURE);
	}

	if (!S_ISCHR(st.st_mode)) {
		fprintf(stderr, "%s is no device\n", dev_name);
		exit(EXIT_FAILURE);
	}

	fd = v4l2_open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

	if (-1 == fd) {
		fprintf(stderr, "Cannot open '%s': %d, %s\n",
		        dev_name, errno, strerror(errno));
		exit(EXIT_FAILURE);
	}
	return fd;
}


// void enum_frameformats(int fd){

// 	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     struct v4l2_fmtdesc fmt;
//     struct v4l2_frmsizeenum frmsize;

//     fmt.index = 0;
//     fmt.type = type;
//     while (xioctl(fd, VIDIOC_ENUM_FMT, &fmt) >= 0) {
//         frmsize.pixel_format = fmt.pixelformat;
//         char c[4];
//         c[0] =  (char) (fmt.pixelformat>>0);
//         c[1] =  (char) (fmt.pixelformat>>8);
//         c[2] =  (char) (fmt.pixelformat>>16);
//         c[3] =  (char) (fmt.pixelformat>>24);
//         printf("Pixelformat %s \n", c);
//         frmsize.index = 0;
//         while (xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) >= 0) {
//             if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
//                 printf("%dx%d\n",
//                                   frmsize.discrete.width,
//                                   frmsize.discrete.height);
//             } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
//                 printf("%dx%d\n",
//                                   frmsize.stepwise.max_width,
//                                   frmsize.stepwise.max_height);
//             }
//                 frmsize.index++;
//             }
//             fmt.index++;
//     }
// }

// 	open_device();
// 	init_device();
// 	start_capturing();
// 	mainloop();
// 	stop_capturing();
// 	uninit_device();
// 	close_device();
// 	fprintf(stderr, "\n");
// 	return 0;



   // enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
   //  struct v4l2_fmtdesc fmt;
   //  struct v4l2_frmsizeenum frmsize;

   //  fmt.index = 0;
   //  fmt.type = type;
   //  while (ioctl(fd, VIDIOC_ENUM_FMT, &fmt) >= 0) {
   //      frmsize.pixel_format = fmt.pixelformat;
   //      frmsize.index = 0;
   //      while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) >= 0) {
   //          if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
   //              printf("%dx%d\n",
   //                                frmsize.discrete.width,
   //                                frmsize.discrete.height);
   //          } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
   //              printf("%dx%d\n",
   //                                frmsize.stepwise.max_width,
   //                                frmsize.stepwise.max_height);
   //          }
   //              frmsize.index++;
   //          }
   //          fmt.index++;
   //  }