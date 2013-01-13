import cv2 as cv
import numpy as np


def main():
    # change this path to point to the data folder you would like to play
    # data_folder = "/Users/mkassner/Downloads/CATMAI_VDO/youtube_data004"
    # data_folder = "/Users/mkassner/Downloads/CATMAI_VDO/phone_data003"
    # data_folder = "/Users/mkassner/Downloads/02/data004"
    # data_folder = "/Users/mkassner/Downloads/samy_oil/random"


    video_path = data_folder + "/world.avi"
    gaze_positions_path = data_folder + "/gaze_positions.npy"

    cap = cv.VideoCapture(video_path)
    gaze_list = list(np.load(gaze_positions_path))

    # this line takes the gaze list and makes a list
    # with the length of the number of recorded frames.
    # Each slot conains a list that has 0, 1 or more assosiated gaze postions.
    positions_by_frame = [[{'x': s[0], 'y': s[1], 'dt': s[2]} \
                         for s in gaze_list if s[-1] == frame] \
                         for frame in range(int(gaze_list[-1][-1]) + 1)]

    # for elm in positions_by_frame:
    #     print elm

    status, img = cap.read()
    height, width = img.shape[0:2]
    frame = 0
    while status:
        current_gaze = positions_by_frame[frame]
        for gaze_point in current_gaze:
            x_screen, y_screen = denormalize((gaze_point['x'], gaze_point['y']), width, height, flip_y=False)
            cv.circle(img, (x_screen, y_screen), 35, (255, 255, 255), 2, cv.cv.CV_AA)
        cv.imshow("world", img)
        status, img = cap.read()
        frame += 1
        ch = cv.waitKey(60)
        if ch == 27:
            break


def denormalize(pos, width, height, flip_y=True):
    """
    denormalize and return as int
    """
    x = pos[0]
    y = pos[1]
    if flip_y:
        y=-y

    x = (x * width / 2.) + (width / 2.)
    y = (y * height / 2.) + (height / 2.)
    return int(x), int(y)

if __name__ == '__main__':
    main()