import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pykitti
from visual_odometry.kitti_reader import DatasetReaderKITTI


def drawFrameFeatures(frame, prevPts, currPts, frameIdx):
    currFrameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for i in range(len(currPts) - 1):
        cv2.circle(currFrameRGB, tuple(currPts[i].astype(np.int)), radius=3, color=(200, 100, 0))
        cv2.line(currFrameRGB, tuple(prevPts[i].astype(np.int)), tuple(currPts[i].astype(np.int)), color=(200, 100, 0))
        cv2.putText(currFrameRGB, "Frame: {}".format(frameIdx), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200))
        cv2.putText(currFrameRGB, "Features: {}".format(len(currPts)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200))
    cv2.imshow("Frame with keypoints", currFrameRGB)


#
# @param trackedPoints
# @param groundtruthPoints
def updateTrajectoryDrawing(trackedPoints, groundtruthPoints):
    plt.cla()
    plt.plot(trackedPoints[:, 0], trackedPoints[:, 2], c='blue', label="Tracking")
    plt.plot(groundtruthPoints[:, 0], groundtruthPoints[:, 2], c='green', label="Ground truth")
    plt.title("Trajectory")
    plt.legend()
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


def monocular_visual_odometry(data_path):
    dataset_reader = DatasetReaderKITTI(data_path)
    intrinsic_matrix = dataset_reader.readCameraMatrix()
    feature_detector = cv2.GFTTDetector_create()

    prev_points = np.array([])
    prev_frame_3 = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    vo_rotation = np.eye(3)
    vo_position = np.zeros(3)
    plt.show()

    for ii in range(1, 250):
        cur_frame_3 = dataset_reader.readFrame(ii)
        prev_frame = cv2.cvtColor(prev_frame_3, cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.cvtColor(cur_frame_3, cv2.COLOR_BGR2GRAY)

        prev_points = feature_detector.detect(prev_frame)
        prev_points = cv2.KeyPoint_convert(sorted(prev_points, key=lambda p: p.response, reverse=True))

        prev_points, cur_points = find_features_in_cur_frame(prev_frame, cur_frame, prev_points)

        if False:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(prev_frame)
            plt.scatter(prev_points[:, 0], prev_points[:, 1], color='red', marker='+', s=1)
            plt.title('prev frame')

            plt.subplot(2, 1, 2)
            plt.imshow(cur_frame)
            plt.scatter(cur_points[:, 0], cur_points[:, 1], color='red', marker='+', s=1)
            plt.title('cur_points')
            plt.show(block=False)

        # Essential matrix, pose estimation
        E, mask = cv2.findEssentialMat(cur_points, prev_points, intrinsic_matrix, cv2.RANSAC, 0.99, 1.0, None)
        prev_points = prev_points[np.squeeze(mask).astype(bool)]
        cur_points = cur_points[np.squeeze(mask).astype(bool)]

        _, R, T, _ = cv2.recoverPose(E, cur_points, prev_points, intrinsic_matrix)

        gt_position, gt_scale = dataset_reader.readGroundtuthPosition(ii)
        if gt_scale <= 0.1:
            continue
        # gt_scale = 1
        vo_position = vo_position + gt_scale * vo_rotation.dot(T)
        vo_rotation = R.dot(vo_rotation)

        kitti_positions.append(gt_position)
        track_positions.append(vo_position)
        updateTrajectoryDrawing(np.array(track_positions), np.array(kitti_positions))
        drawFrameFeatures(cur_frame, prev_points, cur_points, ii)

        if cv2.waitKey(1) == ord('q'):
            break

        prev_points, prev_frame = cur_points, cur_frame
        prev_frame_3 = cur_frame_3

    cv2.destroyAllWindows()
    a = 3


def main():
    # my_dataset = '00'
    # my_dataset = '01'
    # my_dataset = '02'
    # my_dataset = '03'
    # my_dataset = '04' # bad
    # my_dataset = '05'
    my_dataset = '06' # bad
    # my_dataset = '07'
    # my_dataset = '08' # bad
    # my_dataset = '09'
    # my_dataset = '10'
    data_path = f'/home/nadav/studies/mapping_and_perception_autonomous_robots/kitti_data/visual_odometry/dataset/sequences/{my_dataset}'
    monocular_visual_odometry(data_path)


if __name__ == "__main__":
    np.random.seed(0)
    main()
