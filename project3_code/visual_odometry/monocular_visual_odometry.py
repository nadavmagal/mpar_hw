import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pykitti
from visual_odometry.kitti_reader import DatasetReaderKITTI
from frames_to_video import create_video

CREATE_VIDEO = True
if CREATE_VIDEO:
    frame_fig = plt.figure(figsize=[15, 10])


def show_frame(cur_frame, cur_points, gt_coordinated, ii, prev_points, track_coordinates, result_dir_timed):
    if CREATE_VIDEO:
        plt.figure(frame_fig)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(track_coordinates[:, 0], track_coordinates[:, 2], c='black', label="VO")
        plt.plot(gt_coordinated[:, 0], gt_coordinated[:, 2], c='blue', label="Ground truth")
        plt.title("GT and estimated trajectory")
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend()
        plt.draw()

        plt.subplot(1, 2, 2)
        currFrameRGB = cv2.cvtColor(cur_frame, cv2.COLOR_GRAY2RGB)
        for i in range(len(cur_points) - 1):
            cv2.circle(currFrameRGB, tuple(cur_points[i].astype(np.int)), radius=3, color=(200, 100, 0))
            cv2.line(currFrameRGB, tuple(prev_points[i].astype(np.int)), tuple(cur_points[i].astype(int)),
                     color=(200, 100, 0))
        plt.imshow(currFrameRGB)
        plt.title("image and features")

        os.makedirs(result_dir_timed, exist_ok=True)
        plt.savefig(os.path.join(result_dir_timed, f'{ii}.png'), dpi=150)
    else:
        updateTrajectoryDrawing(track_coordinates, gt_coordinated)
        drawFrameFeatures(cur_frame, prev_points, cur_points, ii)


def drawFrameFeatures(frame, prev_points, cur_points, frame_index):
    currFrameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for i in range(len(cur_points) - 1):
        cv2.circle(currFrameRGB, tuple(cur_points[i].astype(np.int)), radius=3, color=(200, 100, 0))
        cv2.line(currFrameRGB, tuple(prev_points[i].astype(np.int)), tuple(cur_points[i].astype(np.int)),
                 color=(200, 100, 0))
        cv2.putText(currFrameRGB, "Frame: {}".format(frame_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200))
        cv2.putText(currFrameRGB, "Features: {}".format(len(cur_points)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200))
    cv2.imshow("Frame with keypoints", currFrameRGB)


#
# @param trackedPoints
# @param groundtruthPoints
def updateTrajectoryDrawing(track_coordinates, gt_coordinated):
    plt.cla()
    plt.plot(track_coordinates[:, 0], track_coordinates[:, 2], c='blue', label="Tracking")
    plt.plot(gt_coordinated[:, 0], gt_coordinated[:, 2], c='green', label="Ground truth")
    plt.title("Trajectory")
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.draw()
    plt.pause(0.01)


def find_bad_features_indices(features, frame, status):
    status_ = status.copy()
    for ii, point in enumerate(features):
        if point[0] < 0 or point[1] < 0 or point[0] > frame.shape[1] or point[1] > frame.shape[0]:
            status_[ii] = 0
    missing_previous_points_indices = np.where(status_ == 0)[0]
    return missing_previous_points_indices


def find_features_in_cur_frame(prev_frame, cur_frame, prev_points):
    cur_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame, prev_points, None)

    bad_indices = find_bad_features_indices(cur_points, cur_frame, status)
    prev_points = np.delete(prev_points, bad_indices, axis=0)
    cur_points = np.delete(cur_points, bad_indices, axis=0)

    return prev_points, cur_points


def monocular_visual_odometry(data_path, result_dir_timed):
    dataset_reader = DatasetReaderKITTI(data_path)
    intrinsic_matrix = dataset_reader.readCameraMatrix()
    feature_detector = cv2.GFTTDetector_create()

    prev_points = np.array([])
    prev_frame_3 = dataset_reader.readFrame(0)
    gt_coordinated, track_coordinates = [], []
    vo_rotation = np.eye(3)
    vo_position = np.zeros(3)
    plt.figure()
    # plt.show()

    for ii in range(1, dataset_reader.getNumberFrames()):
        cur_frame_3 = dataset_reader.readFrame(ii)
        prev_frame = cv2.cvtColor(prev_frame_3, cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.cvtColor(cur_frame_3, cv2.COLOR_BGR2GRAY)

        if False:
            prev_points = feature_detector.detect(prev_frame)
            prev_points = cv2.KeyPoint_convert(sorted(prev_points, key=lambda p: p.response, reverse=True))
            prev_points = prev_points[0:500, :]
        else:
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(prev_frame, None)
            prev_points = cv2.KeyPoint_convert(sorted(kp, key=lambda p: p.response, reverse=True))

        prev_points, cur_points = find_features_in_cur_frame(prev_frame, cur_frame, prev_points)

        if False:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(prev_frame, cmap='gray')
            plt.scatter(prev_points[:, 0], prev_points[:, 1], color='red', marker='+', s=1)
            plt.title('prev frame')

            plt.subplot(2, 1, 2)
            plt.imshow(cur_frame, cmap='gray')
            plt.scatter(cur_points[:, 0], cur_points[:, 1], color='red', marker='+', s=1)
            plt.title('cur_points')
            plt.show(block=False)

        E, mask = cv2.findEssentialMat(cur_points, prev_points, intrinsic_matrix, cv2.RANSAC, 0.99, 1.0, None)
        prev_points = prev_points[np.squeeze(mask).astype(bool)]
        cur_points = cur_points[np.squeeze(mask).astype(bool)]

        _, R, T, _ = cv2.recoverPose(E, cur_points, prev_points, intrinsic_matrix)

        gt_position, dl = dataset_reader.readGroundtuthPosition(ii)

        vo_position = vo_position + np.squeeze((dl * vo_rotation @ T).T)

        vo_rotation = R.dot(vo_rotation)

        gt_coordinated.append(gt_position)
        track_coordinates.append(vo_position)
        show_frame(cur_frame, cur_points, np.array(gt_coordinated), ii, prev_points, np.array(track_coordinates),
                   result_dir_timed)

        if cv2.waitKey(1) == ord('q'):
            break

        prev_points, prev_frame = cur_points, cur_frame
        prev_frame_3 = cur_frame_3

    gt_coordinated = np.array(gt_coordinated)
    track_coordinates = np.array(track_coordinates)

    ex = track_coordinates[:, 0] - gt_coordinated[:, 0]
    ey = track_coordinates[:, 2] - gt_coordinated[:, 2]
    RMSE = np.sqrt(np.mean(np.power(ex, 2) + np.power(ey, 2)))

    print(f'RMSE={RMSE}')

    cv2.destroyAllWindows()
    plt.show()
    a = 3


def main():
    # my_dataset = '00'
    # my_dataset = '01'
    # my_dataset = '02'
    # my_dataset = '03'
    # my_dataset = '04' # bad
    # my_dataset = '05'
    my_dataset = '06'  # bad
    # my_dataset = '07'
    # my_dataset = '08' # bad
    # my_dataset = '09'
    # my_dataset = '10'
    data_path = f'/home/nadav/studies/mapping_and_perception_autonomous_robots/kitti_data/visual_odometry/dataset/sequences/{my_dataset}'

    result_dir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_3/results/vo'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")

    result_dir_timed = os.path.join(result_dir, f'{cur_date_time}')
    print(f'saving to: {result_dir_timed}')

    monocular_visual_odometry(data_path, result_dir_timed)
    if CREATE_VIDEO:
        create_video(result_dir_timed)


if __name__ == "__main__":
    np.random.seed(0)
    main()
