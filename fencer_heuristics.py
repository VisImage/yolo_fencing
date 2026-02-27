# importing libraries
import collections
import json
import math
import os
from collections import defaultdict

import cv2

# import cv2
import numpy as np

# #coco output format
# Neck_idx        = 1
# RShoulder_idx   = 2
# RElbow_idx      = 3
# RWrist_idx      = 4
# LShoulder_idx   = 5
# LElbow_idx      = 6
# LWrist_idx      = 7
# RHip_idx        = 8
# RKnee_idx       = 9
# RAnkle_idx      = 10
# LHip_idx        = 11
# LKnee_idx       = 12
# LAnkle_idx      = 13


#
LShoulder_idx = 5
RShoulder_idx = 6
LElbow_idx = 7
RElbow_idx = 8
LWrist_idx = 9
RWrist_idx = 10
LHip_idx = 11
RHip_idx = 12
LKnee_idx = 13
RKnee_idx = 14
LAnkle_idx = 15
RAnkle_idx = 16


# leftFencer_LElbow = joints_1[7]
# leftFencer_RElbow = joints_1[8]
# leftFencer_LWrist = joints_1[9]
# leftFencer_RWrist = joints_1[10]
# leftFencer_LShoulder = joints_1[5]
# leftFencer_RShoulder = joints_1[6]


WHITE = (255, 255, 255)
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


# json_obj_list: the personal json object in one frame, input
# img: the image of the frame, input and output
def feet_visible(json_obj, conf_th=0.25):
    """Returns True only if BOTH ankles are visible with sufficient confidence. COCO-17: 15 = left ankle 16 = right
    ankle.
    """
    k = json_obj.get("keypoints", [])
    if len(k) < 17 * 3:
        return False

    la_conf = float(k[15 * 3 + 2])
    ra_conf = float(k[16 * 3 + 2])

    return (la_conf >= conf_th) and (ra_conf >= conf_th)


def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)


def drawPose(json_obj_list, img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.mean(img_gray)
    img_out = img

    for json_obj in json_obj_list:
        # if frame_id > 0 :
        keypoints = json_obj["keypoints"]
        joints = []
        for i in range(int(len(keypoints) / 3)):
            joint = (keypoints[i * 3], keypoints[i * 3 + 1])
            joints.append(joint)
        # print(f'keypoints:{joints}')
        # cPts = np.array([joints[2], joints[5], joints[12], joints[9],joints[2]])
        cPts = np.array([joints[RShoulder_idx], joints[RElbow_idx], joints[RWrist_idx]])
        cv2.polylines(img_out, [cPts.astype(int)], False, (255, 0, 0), 2)
        cPts = np.array([joints[LShoulder_idx], joints[LElbow_idx], joints[LWrist_idx]])
        cv2.polylines(img_out, [cPts.astype(int)], False, (255, 0, 0), 1)

        cPts = np.array([joints[RHip_idx], joints[RKnee_idx], joints[RAnkle_idx]])
        cv2.polylines(img_out, [cPts.astype(int)], False, (255, 255, 0), 2)
        cPts = np.array([joints[LHip_idx], joints[LKnee_idx], joints[LAnkle_idx]])
        cv2.polylines(img_out, [cPts.astype(int)], False, (255, 255, 0), 1)


def drawPoseBBox(pose_list, img):
    img_out = img

    for pose in pose_list:
        bbox = pose["box"]
        bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]  # xmin,xmax,ymin,ymax

        cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), WHITE, 1)
        # if opt.tracking:
        cv2.putText(
            img, str(pose["idx"]), (int(bbox[0]), int(bbox[2] + 26)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1
        )

    return img_out


# change idx to string of numbers
def normaliseIdx(pose_list):
    list_out = []
    for pose in pose_list:
        idx = pose["idx"]
        if isinstance(idx, str):
            idx_out = idx
        else:
            idx_out = str(int(idx))
        pose_out = pose
        pose_out["idx"] = [idx_out]
        list_out.append(pose_out)
    return list_out


# Video Generating function
def generate_video(image_folder, video_name):
    # image_folder = '.' # make sure to use your folder
    # video_name = 'mygeneratedvideo.avi'
    # os.chdir("C:\\Python\\Geekfolder2")

    images = [
        img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")
    ]

    # Array images should only consider
    # the image files ignoring others if any
    images.sort()
    # print(f'ImageSize={len(images)}; video_name = {video_name}  images= {images}')

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, _layers = frame.shape
    print(f"h={height}; w={width}")
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    # video = cv.VideoWriter(file_path, fourcc, fps, (w, h))

    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    # Appending the images to the video one by one
    for image in images:
        # fileName = os.path.join(image_folder, image)
        # print(f'fileName={fileName}')
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # print(images)
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated
    print(f"{video_name} is generated.")


def get_frame_dict(pose_list):
    frame_dict = defaultdict()
    for pose in pose_list:
        frame_id = pose["image_id"]
        if frame_id in frame_dict:
            IV = int(pose["idx"])
            frame_dict[frame_id].append(IV)
        else:
            frame_dict[frame_id] = [int(pose["idx"])]
    return frame_dict


# generate pose dictionary
def get_pose_dict(pose_list):

    frame_dict = get_frame_dict(pose_list)
    # for pose in pose_list:
    #     frame_id = pose["image_id"]
    #     if frame_id in frame_dict:
    #         IV = int(pose["idx"])
    #         frame_dict[frame_id].append(IV)
    #     else:
    #         frame_dict[frame_id] = [int(pose["idx"])]

    pose_dict = defaultdict()
    for pose in pose_list:
        image_id = int(pose["image_id"].replace(".jpg", ""))
        pose_id = pose["idx"]
        pose_id = str(int(float(pose_id)))  # with --detector tracker  "idx": 9.0
        # print( "pose_id 2 =", pose_id)
        if pose_id == "765" and image_id == 7:
            pass
        if pose_id == "1153" and image_id == 2328:
            pass
        if pose_id in pose_dict:
            pose_list = pose_dict[pose_id]
            frame_id = str(image_id - 1) + ".jpg"
            if frame_id in frame_dict:
                if int(pose_id) in frame_dict[str(image_id - 1) + ".jpg"]:
                    pose_list[len(pose_list) - 1][1] = image_id
                else:
                    pose_list.append([image_id, image_id])
            else:
                pose_list.append([image_id, image_id])
        else:
            pose_dict[pose_id] = [[image_id, image_id]]

    return pose_dict


# generate pose dictionary
def get_pose_dict_2(pose_list):

    pose_dict = defaultdict()
    for pose in pose_list:
        image_id = int(pose["image_id"].replace(".jpg", ""))
        for pose_id in pose["idx"]:
            # pose_id = pose['idx']
            # pose_id = str(int(float(pose_id)))  # with --detector tracker  "idx": 9.0
            if pose_id in pose_dict:
                pose_dict[pose_id][1].append(image_id)
            else:
                pose_dict[pose_id] = [pose["idx"], [image_id]]

        # frame_dict_name = result +"/frame_dict.txt"
        # with open(frame_dict_name, 'w') as f:
        #     print(frame_dict, file=f)

        # pose_dict_name = result +"/pose_dict.txt"
        # with open(pose_dict_name, 'w') as f:
        #     print(pose_dict, file=f)

    return pose_dict


# generate pose dictionary
def clean_merged_pose(pose_dict):

    pose_dict_temp1 = pose_dict.copy()
    merged_list_final = []
    defaultdict()
    while len(pose_dict_temp1) > 0:
        # take the first element in the dictionary
        # and compare
        key1 = next(iter(pose_dict_temp1.keys()))
        item1 = pose_dict_temp1[key1]
        merged_list_id = item1[0]
        merged_list_imageId = item1[1]
        pose_dict_temp1.pop(key1)
        pose_dict_temp = pose_dict_temp1.copy()
        for pose_id in pose_dict_temp1:
            item = pose_dict_temp1[pose_id]
            common_list = set(item[0]).intersection(item1[0])
            if len(common_list) > 0:
                pose_dict_temp.pop(pose_id)
                merged_list_id = merged_list_id + item[0]
                merged_list_imageId = merged_list_imageId + item[1]
        merged_list_id = list(dict.fromkeys(merged_list_id))
        merged_list_final.append([merged_list_id, merged_list_imageId])
        # merged_dict[merged_list_id] = merged_list_imageId
        pose_dict_temp1 = pose_dict_temp

    return merged_list_final


def get_overlap_dict(frame_dict):
    overlap_dict = defaultdict()
    for i in range(len(frame_dict)):
        pose_list = frame_dict[i]
        list_len = len(pose_list)
        for j in range(list_len):
            box_j = pose_list[j]["box"]
            for k in range(j + 1, list_len):
                box_k = pose_list[k]["box"]
                over_lap_rate = float(int(overLap_rate(box_j, box_k) * 1000)) / 1000
                if over_lap_rate > 0:
                    overlap_dict[i] = [j, k, over_lap_rate]

    return overlap_dict


def removeDarkClothing(json_obj_list, img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(img_gray)
    json_list_output = []

    for json_obj in json_obj_list:
        # if frame_id > 0 :
        keypoints = json_obj["keypoints"]
        joints = []
        for i in range(int(len(keypoints) / 3)):
            joint = (keypoints[i * 3], keypoints[i * 3 + 1])
            joints.append(joint)
        # print(f'keypoints:{joints}')
        # cPts = np.array([joints[2], joints[5], joints[12], joints[9],joints[2]])
        cPts = np.array([joints[5], joints[6], joints[12], joints[11], joints[5]])
        maskImage = np.zeros_like(img_gray)
        cv2.drawContours(maskImage, [cPts.astype(int)], 0, 255, -1)
        local_mean = cv2.mean(img_gray, mask=maskImage)

        if mean < local_mean:  # fencing clothing is white so the brightness value is high
            json_list_output.append(json_obj)

    return json_list_output


# json_obj_list: the personal json object in one frame, input
# img: the image of the frame, input and output


def removeFlat(json_obj_list, img):
    #    remove pose a flat width > 2 * height bbox
    json_list_output = []

    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        box = json_obj["box"]
        width = box[2]
        height = box[3]
        if width < height * 2:
            json_list_output.append(json_obj)
    img_out = img
    return json_list_output, img_out


def removeFlat_2(json_obj_list):
    #    remove pose a flat width > 2 * height bbox
    json_list_output = []

    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        box = json_obj["box"]
        width = box[2]
        height = box[3]
        if width < height * 2:
            json_list_output.append(json_obj)
    return json_list_output


def removeSmall_2(json_obj_list):
    #    if image_id > 285:
    #        print(f'image_id = {image_id}, list_len={json_obj_list}')
    json_list_output = []
    # sort list by box area
    # index = len(json_obj_list) - 1
    while len(json_obj_list) > 0:
        max_obj_area = 0
        for i in range(len(json_obj_list)):
            json_obj = json_obj_list[i]
            box = json_obj["box"]
            if box[2] * box[3] > max_obj_area:
                II = i
                max_obj_area = box[2] * box[3]
        json_list_output.append(json_obj_list[II])
        json_obj_list.pop(II)

    json_obj_list = json_list_output

    json_list_output = []
    max_box = json_obj_list[0]["box"]
    max_obj_area = max_box[2] * max_box[3]

    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        box = json_obj["box"]
        p1 = [box[0], box[1]]
        p2 = [box[0], box[1] + box[3]]
        p3 = [box[0] + box[2], box[1] + box[3]]
        p4 = [box[0] + box[2], box[1]]
        np.array([p1, p2, p3, p4], np.int32)
        if box[2] * box[3] > max_obj_area / 9:
            json_list_output.append(json_obj)

            json_obj["image_id"]
            # print(f'image_id={str_id}')
            # if str_id == "130.jpg":
            # print(json_obj)
            # sys.exit()
        # else:
        #    cv2.polylines(img_out,[pts],True,(255,0,0), 1)

    return json_list_output


def removeSmall(json_obj_list, img):
    #    if image_id > 285:
    #        print(f'image_id = {image_id}, list_len={json_obj_list}')
    json_list_output = []
    # sort list by box area
    # index = len(json_obj_list) - 1
    while len(json_obj_list) > 0:
        max_obj_area = 0
        for i in range(len(json_obj_list)):
            json_obj = json_obj_list[i]
            box = json_obj["box"]
            if box[2] * box[3] > max_obj_area:
                II = i
                max_obj_area = box[2] * box[3]
        json_list_output.append(json_obj_list[II])
        json_obj_list.pop(II)

    img_out = img
    json_obj_list = json_list_output

    json_list_output = []
    max_box = json_obj_list[0]["box"]
    max_obj_area = max_box[2] * max_box[3]

    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        box = json_obj["box"]
        p1 = [box[0], box[1]]
        p2 = [box[0], box[1] + box[3]]
        p3 = [box[0] + box[2], box[1] + box[3]]
        p4 = [box[0] + box[2], box[1]]
        pts = np.array([p1, p2, p3, p4], np.int32)
        if box[2] * box[3] > max_obj_area / 9:
            json_list_output.append(json_obj)
            cv2.polylines(img_out, [pts], True, (128, 128, 128), 1)
            if int(json_obj["idx"]) == 765:
                cv2.polylines(img_out, [pts], True, (255, 0, 0), 3)
            if int(json_obj["idx"]) == 816:
                cv2.polylines(img_out, [pts], True, (0, 255, 0), 3)
            if int(json_obj["idx"]) == 1016:
                cv2.polylines(img_out, [pts], True, (0, 0, 255), 3)

            # cv2.putText(img_out, f'{V1}-{V2}', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            cv2.putText(
                img_out,
                str(json_obj["idx"]),
                (int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            json_obj["image_id"]
            # print(f'image_id={str_id}')
            # if str_id == "130.jpg":
            # print(json_obj)
            # sys.exit()
        # else:
        #    cv2.polylines(img_out,[pts],True,(255,0,0), 1)

    return json_list_output, img_out


def overLap_rate(b1, b2):
    left = max(b1[0], b2[0])
    right = min(b1[0] + b1[2], b2[0] + b2[2])
    bottom = max(b1[1], b2[1])
    top = min(b1[1] + b1[3], b2[1] + b2[3])
    if left < right and bottom < top:
        area1 = (right - left) * (top - bottom)
        area2 = b1[2] * b1[3]
        area3 = b2[2] * b2[3]
        return area1 / min(area2, area3)
    else:
        return 0


def _get_bbox(keypoints_merge):
    Xmin = 1000000000
    Xmax = 0
    Ymin = 1000000000
    Ymax = 0
    size = int(len(keypoints_merge) / 3)
    for i in range(size):
        x = keypoints_merge[i * 3]
        y = keypoints_merge[i * 3 + 1]
        if Xmin > x:
            Xmin = x
        elif Xmax < x:
            Xmax = x
        if Ymin > y:
            Ymin = y
        elif Ymax < y:
            Ymax = y
    return Xmin, Ymin, Xmax - Xmin, Ymax - Ymin


def get_Overlap_list(json_obj_list):
    overlap_list = []
    no_pose_in_frame = len(json_obj_list)
    for j in range(no_pose_in_frame):
        Jbox = json_obj_list[j]["box"]
        for k in range(j + 1, no_pose_in_frame):
            over_lap_rate = float(int(overLap_rate(Jbox, json_obj_list[k]["box"]) * 1000)) / 1000
            if over_lap_rate > 0:
                overlap_list.append([json_obj_list[j]["idx"], json_obj_list[k]["idx"], over_lap_rate])

    return overlap_list


def remove_overlap_by_size(json_obj_list, overlap_list, ratio):

    for i in range(len(overlap_list)):
        overlap = overlap_list[i]

        box_idx_0 = 0
        box_idx_1 = 0
        json_list_out = []
        over_lap_out = []
        for pose in json_obj_list:
            if pose["idx"] == overlap[0]:
                box_idx_0 = pose["idx"]
                box_area_0 = pose["box"][2] * pose["box"][3]
            elif pose["idx"] == overlap[1]:
                box_idx_1 = pose["idx"]
                box_area_1 = pose["box"][2] * pose["box"][3]
        if box_area_0 * ratio < box_area_1:
            for pose in json_obj_list:
                if pose["idx"] != box_idx_0:
                    json_list_out.append(pose)
        elif box_area_1 * ratio < box_area_0:
            for pose in json_obj_list:
                if pose["idx"] != box_idx_1:
                    json_list_out.append(pose)
        else:
            over_lap_out.append(overlap)
            for pose in json_obj_list:
                json_list_out.append(pose)

    return json_list_out, over_lap_out


def keypoint_dis(list1, list2):
    return 0


def keypoint_average(list1, list2):
    return list1


def get_diff(list1):
    return list1


def merge_pose_list(list1, list2):
    return list1


def merge_overlap_pose(pose_list, overlap_list):
    overlap_out = []
    pose_out = []
    for overlap in overlap_list:
        idx0 = overlap[0]
        idx1 = overlap[1]
        for pose in pose_list:
            if pose["idx"] == idx0:
                pose_0 = pose
            elif pose["idx"] == idx1:
                pose_1 = pose
            else:
                pose_out.append(pose)

        keypoints_0 = pose_0["keypoints"]
        keypoints_1 = pose_1["keypoints"]
        size = int(len(keypoints_0) / 3)
        keypoints_diff = [0] * size
        pose_merge = pose_0
        keypoints_merge = keypoints_0
        for index in range(size):
            i = index * 3
            score_thre = 0.6
            keypoints_merge[i] = (keypoints_0[i] + keypoints_1[i]) / 2
            keypoints_merge[i + 1] = (keypoints_0[i + 1] + keypoints_1[i + 1]) / 2
            keypoints_merge[i + 2] = min(keypoints_0[i + 2], keypoints_1[i + 2])

            if keypoints_0[i + 2] > score_thre and keypoints_1[i + 2] > score_thre:
                keypoints_diff[index] = max(
                    (keypoints_0[i] - keypoints_1[i]), (keypoints_0[i + 1] - keypoints_1[i + 1])
                )

        merge_thre = 15
        merge_status = True
        for diff in keypoints_diff:
            if diff > merge_thre:
                merge_status = False

        if merge_status:
            if pose_0["score"] > pose_1["score"]:
                pose_merge = pose_0
            else:
                pose_merge = pose_1
            pose_merge["keypoints"] = keypoints_merge
            pose_merge["score"] = max(pose_0["score"], pose_1["score"])
            pose_merge["box"] = _get_bbox(keypoints_merge)
            pose_merge["idx"] = pose_0["idx"] + pose_1["idx"]
            pose_out.append(pose_merge)
        else:
            overlap_out.append(overlap)
            pose_out.append(pose_0)
            pose_out.append(pose_1)

    return pose_out, overlap_out


def remove_overlap_by_merge(pose_list, thre):
    pose_list_out = []
    length = len(pose_dict)
    for i in range(length):
        list_i = pose_dict[i]
        for j in range(i + 1, length):
            list_j = pose_dict[j]
            comman_frame_list = list_i & list_j
            if comman_frame_list != []:
                continue
        if comman_frame_list == []:
            pose_list_out.append(list_i)
        else:
            comman_frame_list = list_i & list_j
            get_diff(comman_frame_list)
            # if comman_frame_list == []
            #     pose_list_out.append(merge_pose_list(list_i, list_j)
            # pose_1 = -1
            # pose_list_out = []
        #     for pose in json_obj_list:
        #         if pose['idx'] == overlap[0]:
        #             box_idx_0 = pose['idx']
        #             keypoints_0 = pose['keypoints']
        #         elif pose['idx'] == overlap[1]:
        #             box_idx_1 = pose['idx']
        #             keypoints_1 = pose['keypoints']
        #     if keypoint_dis (keypoints_0,keypoints_1) < thre:
        #         for pose in json_obj_list:
        #             if pose['idx'] != box_idx_0:
        #                 json_list_out.append(pose)        json_list_out = []
        # over_lap_out = []
        # else:
        #     over_lap_out.append(overlap)

    return pose_list_out


def remove_overlap_by_sequence(pose_list, overlap_list, pose_dict, frame_list):
    pose_list_out = []
    pose_dict_out = []

    for i in range(len(overlap_list)):
        overlap_list[i]

    return pose_list_out, pose_dict_out


def _arm_forward_ok(json_obj, conf_th=0.25) -> bool:
    """True if at least one arm (elbow->wrist) is mostly horizontal (forward), with both keypoints present. COCO-17
    indices used by your file: 7=L elbow, 8=R elbow, 9=L wrist, 10=R wrist.
    """
    k = json_obj.get("keypoints", [])
    if len(k) < 17 * 3:
        return False

    def kp(i):
        return (float(k[i * 3 + 0]), float(k[i * 3 + 1]), float(k[i * 3 + 2]))

    # left arm
    ex, ey, ec = kp(7)
    wx, wy, wc = kp(9)
    left_ok = ec >= conf_th and wc >= conf_th and abs(ey - wy) < abs(ex - wx)

    # right arm
    ex, ey, ec = kp(8)
    wx, wy, wc = kp(10)
    right_ok = ec >= conf_th and wc >= conf_th and abs(ey - wy) < abs(ex - wx)

    return left_ok or right_ok


def _feet_sep_ok(json_obj) -> bool:
    """True if ankle-to-ankle distance is large enough relative to bbox height. This is from the intent in
    PostProcessing.py / older heuristic.
    """
    box = json_obj.get("box", None)
    k = json_obj.get("keypoints", [])
    if box is None or len(k) < 17 * 3:
        return False

    # ankles: 15,16
    ax1, ay1 = float(k[15 * 3 + 0]), float(k[15 * 3 + 1])
    ax2, ay2 = float(k[16 * 3 + 0]), float(k[16 * 3 + 1])

    h = float(box[3]) if len(box) >= 4 else 0.0
    if h <= 1e-6:
        return False

    return distance((ax1, ay1), (ax2, ay2)) > (h / 4.0)


def is_mostly_white(frame_bgr, json_obj, margin=0.0):
    """Decide whether the person is wearing mostly-white fencing clothing.

    This replaces the old bbox/HSV heuristic with the same core idea used by `removeDarkClothing()` in
    PostProcessing.py: compare brightness of the *torso area* (shoulders + hips polygon) against the global frame
    brightness.

    Returns True if the torso region is brighter than the overall frame by at least `margin`.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full BGR frame.
    json_obj : dict
        A single AlphaPose-style person object containing "keypoints" and "box".
    margin : float
        Optional extra margin (in grayscale intensity units) required for torso
        to be considered "white".
    """
    img_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    global_mean = float(cv2.mean(img_gray)[0])

    k = json_obj.get("keypoints", [])
    if len(k) < 17 * 3:
        return False

    # Build joints list [(x,y), ...] from flat keypoints
    joints = []
    for i in range(int(len(k) / 3)):
        joints.append((float(k[i * 3]), float(k[i * 3 + 1])))

    # Torso polygon: LShoulder -> RShoulder -> RHip -> LHip -> LShoulder
    try:
        cPts = np.array(
            [
                joints[LShoulder_idx],
                joints[RShoulder_idx],
                joints[RHip_idx],
                joints[LHip_idx],
                joints[LShoulder_idx],
            ],
            dtype=np.int32,
        )
    except Exception:
        return False

    # If any point is missing/invalid, do not accept
    if cPts.size == 0:
        return False

    mask = np.zeros_like(img_gray, dtype=np.uint8)
    cv2.drawContours(mask, [cPts], 0, 255, -1)

    local_mean = float(cv2.mean(img_gray, mask=mask)[0])

    # fencing clothing is white so the brightness value is high
    return local_mean >= (global_mean + float(margin))


def checkForFencer(json_obj, frame_bgr, return_reason: bool = False):
    """Determine whether a pose json_obj corresponds to a fencer.

    Logic (intended):
    - MUST have feet visible
    - MUST pass mostly-white uniform check
    - AND must satisfy at least one of:
          (a) arm-forward heuristic
          (b) feet-separation heuristic

    Backward compatible:
    - return_reason=False (default) -> returns bool
    - return_reason=True  -> returns (bool, reason_str)

    reason_str is a short token suitable for drawing on debug frames.
    """

    def _ret(ok: bool, reason: str):
        return (ok, reason) if return_reason else ok

    # 1) Mandatory: feet visible
    if not feet_visible(json_obj):
        return _ret(False, "feet_not_visible")

    # 2) Mandatory: mostly-white uniform (frame-dependent)
    # If frame is missing, do not hard-fail; treat as unknown and pass.
    if frame_bgr is not None:
        try:
            if not is_mostly_white(frame_bgr, json_obj):
                return _ret(False, "not_white")
        except Exception:
            # Be conservative: if the color check errors, annotate it explicitly.
            return _ret(False, "white_check_err")

    # 3) At least one of the heuristics must pass
    feet_sep_ok = _feet_sep_ok(json_obj)
    arm_fwd_ok = _arm_forward_ok(json_obj)

    if not (feet_sep_ok or arm_fwd_ok):
        return _ret(False, "feet_sep&arm_fwd_fail")

    # Pass
    return _ret(True, "pass")


def checkForFencer_with_reason(json_obj, frame_bgr):
    """Convenience wrapper: always returns (ok, reason)."""
    return checkForFencer(json_obj, frame_bgr, return_reason=True)


def findCloseKeypoint(json_list_output):
    obj1 = json_list_output[0]
    obj2 = json_list_output[1]

    keypoint1 = obj1["keypoints"]
    keypoint2 = obj2["keypoints"]

    numOfPoint = int(len(keypoint2) / 3)
    joints_1 = []
    joints_2 = []
    for i in range(numOfPoint):
        joint_1 = (keypoint1[i * 3], keypoint1[i * 3 + 1])
        joints_1.append(joint_1)
        joint_2 = (keypoint2[i * 3], keypoint2[i * 3 + 1])
        joints_2.append(joint_2)

    average_1_x = joints_1[-1][0]
    average_1_y = joints_1[-1][1]
    average_2_x = joints_2[-1][0]
    average_2_y = joints_2[-1][1]

    for i in range(numOfPoint - 1):
        # print(f'joints_1[i]:{joints_1[i]}')
        average_1_x += joints_1[i][0]
        average_1_y += joints_1[i][1]
        average_2_x += joints_2[i][0]
        average_2_y += joints_2[i][1]

    average_1_x = average_1_x / numOfPoint
    average_1_y = average_1_y / numOfPoint
    average_2_x = average_2_x / numOfPoint
    average_2_y = average_2_y / numOfPoint

    min_1 = 100000
    min_2 = 100000
    min_1_id = -1
    min_2_id = -1

    average_1 = (int(average_1_x), int(average_1_y))
    average_2 = (int(average_2_x), int(average_2_y))
    for i in range(numOfPoint):
        # print(f'distance(average_2, joints_1[{i}])={distance(average_2, joints_1[i])}')
        if min_1 > distance(average_2, joints_1[i]):
            min_1 = distance(average_2, joints_1[i])
            min_1_id = i
        # print(f'distance(average_1, joints_2[{i}])={distance(average_1, joints_2[i])}')
        if min_2 > distance(average_1, joints_2[i]):
            min_2 = distance(average_1, joints_2[i])
            min_2_id = i

    return min_1_id, min_2_id, min_1, min_2, average_1, average_2


def findMinDisKeypoint(json_list_output):

    keypoint1 = json_list_output[0]["keypoints"]
    keypoint2 = json_list_output[1]["keypoints"]

    numOfPoint = int(len(keypoint2) / 3)
    joints_1 = []
    joints_2 = []
    for i in range(numOfPoint):
        joint_1 = [keypoint1[i * 3], keypoint1[i * 3 + 1]]
        joints_1.append(joint_1)
        joint_2 = [keypoint2[i * 3], keypoint2[i * 3 + 1]]
        joints_2.append(joint_2)

    min_dis = 100000
    min_idx = [-1, -1]
    for i in range(numOfPoint):
        joint1 = joints_1[i]
        for j in range(numOfPoint):
            joint2 = joints_2[j]
            dis = distance(joint1, joint2)
            if dis < min_dis:
                min_dis = dis
                min_idx = [i, j]

    return min_idx, min_dis


def kneeDirectionCheck(Knee, Hip, Ankle):
    # point on the line (x,y)
    # (x - Ankle[0])*(y-Hip[1]) = (x-Hip[0])*(y-Ankle[1])
    # x * (y-Hip[1]) - x * (y-Ankle[1]) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * ((y-Hip[1]) - (y-Ankle[1])) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * (-Hip[1] + Ankle[1]) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * (Ankle[1] -Hip[1] + ) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])

    y = Knee[1]
    if Ankle[1] != Hip[1]:
        x = int((Ankle[0] * (y - Hip[1]) - Hip[0] * (y - Ankle[1])) / (Ankle[1] - Hip[1]))
        if Knee[0] > x:
            return "toRight"
        elif Knee[0] < x:
            return "toLeft"
        else:
            return "unknown"
    else:
        return "unknown"


def elbowDireectionCheck(Knee, Hip, Ankle):
    # point on the line (x,y)
    # (x - Ankle[0])*(y-Hip[1]) = (x-Hip[0])*(y-Ankle[1])
    # x * (y-Hip[1]) - x * (y-Ankle[1]) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * ((y-Hip[1]) - (y-Ankle[1])) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * (-Hip[1] + Ankle[1]) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * (Ankle[1] -Hip[1] + ) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    x = 0
    y = Knee[1]
    if Ankle[1] != Hip[1]:
        x = int((Ankle[0] * (y - Hip[1]) - Hip[0] * (y - Ankle[1])) / (Ankle[1] - Hip[1]))

    if x > 0:
        result = "toLeft"
    elif x < 0:
        result = "toRight"
    else:
        result = "none"
    return result


def check_straightness(p1, p2, p3):
    # A*x + B*y + c = 0
    A = p1[1] - p3[1]
    B = p3[0] - p1[0]
    C = p1[0] * p3[1] - p1[1] * p3[0]
    D_line = abs(A * p2[0] + B * p2[1] + C) / math.sqrt(A * A + B * B)
    Dis = distance(p1, p3)
    R = (Dis - D_line) / Dis

    return R


def getFencerStatus_inPose(pose):

    keypoints = pose["keypoints"]
    numOfPoint = int(len(keypoints) / 3)
    joints = []
    for i in range(numOfPoint):
        joint = (keypoints[i * 3], keypoints[i * 3 + 1])
        joints.append(joint)

    LElbow = joints[7]
    joints[8]
    LWrist = joints[9]
    joints[10]
    joints[10]
    joints[10]
    LHip = joints[15]
    RHip = joints[16]
    LKnee = joints[15]
    RKnee = joints[16]
    LAnkle = joints[15]
    RAnkle = joints[16]

    if checkForFencer(json_obj):
        check_1 = kneeDirectionCheck(LKnee, LHip, LAnkle)
        kneeDirectionCheck(RKnee, RHip, RAnkle)
        check_3 = elbowDireectionCheck(LElbow, LShoulder, LWrist)
        elbowDireectionCheck(LElbow, LShoulder, LWrist)
        if check_1 == "toLeft" and check_3:  # right fencer Knee is toward left
            pass

    return 0
    # status 0: pose un-classified
    # status 1: pose of the fencer on the left
    # status 2: pose of the fencer on the right
    # status 3: pose in dark clothing


def getFencerStatus_inFrame(json_obj_list):
    json_list_output = []

    for pose in json_obj_list:
        getFencerStatus_inPose(pose)
    # print('strp 1================')
    # sort list by box area
    len(json_obj_list) - 1
    while len(json_obj_list) > 0:
        max_obj_area = 0
        for i in range(len(json_obj_list)):
            json_obj = json_obj_list[i]
            box = json_obj["box"]
            if box[2] * box[3] > max_obj_area:
                II = i
                max_obj_area = box[2] * box[3]
        json_list_output.append(json_obj_list[II])


def _get_center(pose):
    x = 0
    y = 0
    size = int(len(pose["keypoints"]) / 3)
    for i in range(size):
        x = x + pose["keypoints"][i * 3]
        y = y + pose["keypoints"][i * 3 + 1]

    return [x / size, y / size]


# 'pelvis', 'left_hip', 'right_hip',      # 2
# 'spine1', 'left_knee', 'right_knee',    # 5
# 'spine2', 'left_ankle', 'right_ankle',  # 8
# 'spine3', 'left_foot', 'right_foot',    # 11
# 'neck', 'left_collar', 'right_collar',  # 14
# 'jaw',                                  # 15
# 'left_shoulder', 'right_shoulder',      # 17
# 'left_elbow', 'right_elbow',            # 19
# 'left_wrist', 'right_wrist',            # 21
# 'left_thumb', 'right_thumb',            # 23
# 'head', 'left_middle', 'right_middle',  # 26
# 'left_bigtoe', 'right_bigtoe'           # 28

# "keypoints": [ "nose", "left_eye", "right_eye", "left_ear", "right_ear",
# "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
# "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
# "left_ankle", "right_ankle" ]


def getFencerPose(pose_list):

    if len(pose_list) < 2:
        return [], []
    dict_bodyLength = defaultdict()
    for pose in pose_list:
        keyP = pose["keypoints"]
        HipL = (int(keyP[11 * 3]), int(keyP[11 * 3 + 1]))
        HipR = (int(keyP[12 * 3]), int(keyP[12 * 3 + 1]))
        KneeL = (int(keyP[13 * 3]), int(keyP[13 * 3 + 1]))
        KneeR = (int(keyP[14 * 3]), int(keyP[14 * 3 + 1]))
        AnkleL = (int(keyP[15 * 3]), int(keyP[15 * 3 + 1]))
        AnkleR = (int(keyP[16 * 3]), int(keyP[16 * 3 + 1]))
        body_length_L = distance(HipL, KneeL) + distance(KneeL, AnkleL)
        body_length_R = distance(HipR, KneeR) + distance(KneeR, AnkleR)
        score_L = min(keyP[11 * 3 + 2], keyP[13 * 3 + 2], keyP[15 * 3 + 2])
        score_R = min(keyP[12 * 3 + 2], keyP[14 * 3 + 2], keyP[16 * 3 + 2])
        if score_L > score_R:
            body_length = body_length_L
        else:
            body_length = body_length_R

        dict_bodyLength[int(body_length)] = pose

    sorted_bodyLength_list = sorted(dict_bodyLength, reverse=True)

    pose_fencer_L = []
    pose_fencer_R = []

    for body_length in sorted_bodyLength_list:
        pose = dict_bodyLength[body_length]
        keyP = pose["keypoints"]

        HipL = (int(keyP[11 * 3]), int(keyP[11 * 3 + 1]))
        HipR = (int(keyP[12 * 3]), int(keyP[12 * 3 + 1]))
        KneeL = (int(keyP[13 * 3]), int(keyP[13 * 3 + 1]))
        KneeR = (int(keyP[14 * 3]), int(keyP[14 * 3 + 1]))
        AnkleL = (int(keyP[15 * 3]), int(keyP[15 * 3 + 1]))
        AnkleR = (int(keyP[16 * 3]), int(keyP[16 * 3 + 1]))

        # WHITE = (255, 255, 255)
        # DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, 'hl', HipL, DEFAULT_FONT, 1, WHITE, 2)
        # cv2.putText(img, 'hr', HipR, DEFAULT_FONT, 1, WHITE, 2)posesorted_dict_list
        result1 = kneeDirectionCheck(KneeL, HipL, AnkleL)
        result2 = kneeDirectionCheck(KneeR, HipR, AnkleR)
        if result1 == "toLeft" or result2 == "toLeft":
            if pose_fencer_R == []:
                pose_fencer_R = pose
            # elif _get_center(pose)[0] > _get_center(pose_fencer_L)[0]: # fencer_L.x is smaller
            #     pose_fencer_R = pose
        elif result1 == "toRight" or result2 == "toRight":
            if pose_fencer_L == []:
                pose_fencer_L = pose
            # elif _get_center(pose)[0] > _get_center(pose_fencer_L)[0]: # fencer_L.x is smaller
            #     pose_fencer_R = pose
    return pose_fencer_L, pose_fencer_R


def get2fencingStatus(json_obj_list, img):
    json_list_output = []
    max_obj_area = 0
    # print('strp 1================')
    # sort list by box area
    len(json_obj_list) - 1
    while len(json_obj_list) > 0:
        max_obj_area = 0
        for i in range(len(json_obj_list)):
            json_obj = json_obj_list[i]
            box = json_obj["box"]
            if box[2] * box[3] > max_obj_area:
                II = i
                max_obj_area = box[2] * box[3]
        json_list_output.append(json_obj_list[II])
        json_obj_list.pop(II)

    # print('strp 2================')
    json_obj_list = json_list_output
    json_list_output = []

    # search for the first fencer
    firstfencerFound = False
    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        if checkForFencer(json_obj):
            firstFencingID = i
            firstfencerFound = True
            box = json_obj["box"]
            cv2.putText(
                img,
                "1st",
                (int(box[0]), int(box[1] + box[3])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            break

    # print('strp 3================')
    if not firstfencerFound:
        img = cv2.putText(
            img, "NOFencer", (img.shape[0] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA
        )
        return img, 0  #

    json_list_output.append(json_obj_list[firstFencingID])
    json_obj_list.pop(firstFencingID)
    secondFencingID = False
    secondfencerFound = False
    if len(json_obj_list) > 0:
        for i in range(len(json_obj_list)):
            json_obj = json_obj_list[i]
            if checkForFencer(json_obj):
                secondfencerFound = True
                box = json_obj["box"]
                cv2.putText(
                    img,
                    "2nd",
                    (int(box[0]), int(box[1] + box[3])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                break

    if not secondfencerFound:
        cv2.putText(img, "NNNNNN", (img.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        return img, 0

    json_list_output.append(json_obj_list[secondFencingID])
    max_1_id, max_2_id, _max_1, _max_2, A1, A2 = findCloseKeypoint(json_list_output)
    cv2.circle(img, A1, 15, (0, 0, 255), -1)
    cv2.circle(img, A2, 15, (0, 0, 255), -1)

    if max_1_id == 9 or (max_1_id == 10 and max_2_id == 9) or max_2_id == 10:
        img = cv2.putText(
            img, "FFFFFF", (img.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA
        )
        return img, 1

    img = cv2.putText(
        img, "UUUUUU", (img.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 128), 4, cv2.LINE_AA
    )
    return img, -1


def get_fencer_list_in_all_frames(frame_dict, fencer_list):
    fencer_list_all_frames = []
    for key in frame_dict:
        fencer_list_in_a_frame = []
        for pose in frame_dict[key]:
            if pose["idx"] in fencer_list:
                fencer_list_in_a_frame.append(pose["idx"])
        fencer_list_all_frames.append(fencer_list_in_a_frame)
    return fencer_list_all_frames


# remove_overlap_fencers: clean poses, which produces false fencer
# 1) mark all the poses as fencer, as long as this pose is detected as fencer once
# 2) remove pose from fencer, which has been both left and right fencer n different frame
# 3) check for duplicated left or right fencers in a single frame, using frame_dict
# 4) remove pose and duplication:
#   i) remove short sequency.  short fencer sequence < long_fencer_sequency /10
#   ii) remove sequency with small bbox if min1 > max2: remove sequency2
# 5) remove the sequences from frame_dict and pose_dict and json_obj_final_list
#


def remove_overlap_fencers(frame_fencer_dict, frame_dict):
    # 1) mark all the poses as fencer, as long as this pose is detected as fencer once
    fencer_left_list = []
    for key, fencer_l in frame_fencer_dict[0].items():
        if fencer_l not in fencer_left_list:
            fencer_left_list.append(fencer_l)
    if "-1" in fencer_left_list:
        fencer_left_list.remove("-1")

    fencer_right_list = []
    for key, fencer_r in frame_fencer_dict[1].items():
        if fencer_r not in fencer_right_list:
            fencer_right_list.append(fencer_r)
    if "-1" in fencer_right_list:
        fencer_right_list.remove("-1")

    # remove pose as the intersect in both left and right fencer lists
    intersect = set(fencer_right_list).intersection(set(fencer_left_list))
    while len(intersect) > 0:
        pose_0 = next(iter(intersect))
        no_a_left = 0
        no_a_right = 0
        for key, fencer_l in frame_fencer_dict[0].items():
            if fencer_l == pose_0:
                no_a_left = no_a_left + 1
        for key, fencer_r in frame_fencer_dict[1].items():
            if fencer_r == pose_0:
                no_a_right = no_a_right + 1

        if no_a_left > no_a_right:
            fencer_right_list.remove(pose_0)
            for key, value in frame_fencer_dict[1].items():
                if value == pose_0:
                    frame_fencer_dict[1][key] = "-1"
        else:
            fencer_left_list.remove(pose_0)
            for key, value in frame_fencer_dict[0].items():
                if value == pose_0:
                    frame_fencer_dict[0][key] = "-1"

        intersect = set(fencer_right_list).intersection(set(fencer_left_list))

    fencer_left_list_all_frames = get_fencer_list_in_all_frames(frame_dict, fencer_left_list)
    fencer_right_list_all_frames = get_fencer_list_in_all_frames(frame_dict, fencer_right_list)

    frameNo = len(frame_dict)

    for id in range(frameNo):
        # process left fencers
        fencer_in_frame = fencer_left_list_all_frames[id]
        size = len(fencer_in_frame)
        if size > 1:
            # no_as = [0]* size
            idx_max = 0
            value_max = -1
            for i in range(size):
                value = collections.Counter(frame_fencer_dict[0])[fencer_in_frame[i]]
                # no_as[i] = value
                if value > value_max:
                    value_max = value
                    idx_max = fencer_in_frame[i]

            for i in range(size):
                pose_idx = fencer_in_frame[i]
                if pose_idx != idx_max:
                    fencer_left_list.remove(pose_idx)
                    for i, value in frame_fencer_dict[0].items():
                        if value == pose_idx:
                            frame_fencer_dict[0][i] = "-1"

            fencer_left_list_all_frames = get_fencer_list_in_all_frames(frame_dict, fencer_left_list)

        # process right fencers
        fencer_in_frame = fencer_right_list_all_frames[id]
        size = len(fencer_in_frame)
        if size > 1:
            idx_max = 0
            value_max = -1
            for i in range(size):
                value = collections.Counter(frame_fencer_dict[1])[fencer_in_frame[i]]
                if value > value_max:
                    value_max = value
                    idx_max = fencer_in_frame[i]

            for i in range(size):
                pose_idx = fencer_in_frame[i]
                if pose_idx != idx_max:
                    fencer_right_list.remove(pose_idx)
                    for i, value in frame_fencer_dict[1].items():
                        if value == pose_idx:
                            frame_fencer_dict[1][i] = "-1"

            fencer_right_list_all_frames = get_fencer_list_in_all_frames(frame_dict, fencer_right_list)

    return frame_fencer_dict


def getIdx(idx_list, pose_list):
    # return the idx found in the existing pose_list
    for pose in pose_list:
        idx = pose["idx"]
        if idx in idx_list:
            return idx
    # return the min id in the list otherwise
    result_idx = 10000
    for idx in idx_list:
        if int(idx) < result_idx:
            result_idx = int(idx)
    return str(result_idx)


# merge pose, re-idx pose which is the same pose but has different idx in
# different frames


def merge_poses(json_obj_final_list):

    pose_list = []
    pose_dict = get_pose_dict_2(json_obj_final_list)

    merged_pose_dict_list = clean_merged_pose(pose_dict)

    for pose in json_obj_final_list:
        for item in merged_pose_dict_list:
            idx_list = item[0]
            if pose["idx"][0] == "266":
                pass
            if pose["idx"][0] in idx_list:
                idx = getIdx(idx_list, pose_list)
                pose["idx"] = idx
                pose_list.append(pose)
                break
    get_pose_dict(pose_list)

    return pose_list


# def get_en_guad_frame_list(fencer_dict):

#     fencer_frame_side_dict = defaultdict()
#     for key in fencer_dict[0]:
#         fencer_list = [fencer_dist[0][key], fencer_dist[1][key]]
#         fencer_frame_side_dict[key] = fencer_list

# en_guad_period = 0.5 #second
# frame_rate = 30
# en_guad_frame = en_guad_period * frame_rate

# left_fencer_arm_straigt = False
# right_fencer_arm_straigt = False
# left_fencer_hand_out = False
# right_fencer_hand_out = False
# still_enough = False

# left_fancer_en_gaud_dist = []
# for key, idx in fencer_dist[0].items():
#     yy = 0

# return fencer_frame_side_dict

# # def get_en_guad_frame_list(fencer_dict, frame_dict):
#     en_guad_frame_dict  = defaultdict()
#     fencer_frame_side_dict = defaultdict()
#     for key in fencer_dict[0]:
#         fencer_list = [fencer_dict[0][key], fencer_dict[1][key]]
#         leftFencerExit = False
#         rightFencerExit = False
#         for pose in frame_dict[key]:
#             if pose['idx'] == fencer_dict[0][key]:
#                 fencer_list[0] = pose
#                 leftFencerExit  = True
#             if pose['idx'] == fencer_dict[1][key]:
#                 fencer_list[1] = pose
#                 rightFencerExit = True
#         if leftFencerExit and rightFencerExit:
#             fencer_frame_side_dict[key] = fencer_list

#     en_guad_frame_list = []
#     for key, list in fencer_frame_side_dict.items():

#         keypoint1 = list[0]['keypoints']
#         keypoint2 = list[1]['keypoints']

#         numOfPoint = int(len(keypoint2)/3)
#         joints_1 = []
#         joints_2 = []
#         for i in range(numOfPoint):
#             joint_1 = [keypoint1[i*3],keypoint1[i*3 + 1]]
#             joints_1.append(joint_1)
#             joint_2 = [keypoint2[i*3],keypoint2[i*3 + 1]]
#             joints_2.append(joint_2)

#         # idx_list, minDis = findMinDisKeypoint(list)
#         # if idx_list[0] == 9 or idx_list[0] == 10:
#         #     if  idx_list[1] == 9 or idx_list[1] == 10:
#         #         hand_in_position = True
#         #     else:
#         #         hand_in_position = False

#         # check if arm is straight forward


#         leftFencer_LElbow = joints_1[7]
#         leftFencer_RElbow = joints_1[8]
#         leftFencer_LWrist = joints_1[9]
#         leftFencer_RWrist = joints_1[10]
#         leftFencer_LShoulder = joints_1[5]
#         leftFencer_RShoulder = joints_1[6]
#         rightFencer_LElbow = joints_2[7]
#         rightFencer_RElbow = joints_2[8]
#         rightFencer_LWrist = joints_2[9]
#         rightFencer_RWrist = joints_2[10]
#         rightFencer_LShoulder = joints_2[5]
#         rightFencer_RShoulder = joints_2[6]

#         #find a shoulder as reference for the other fencer
#         if keypoint1[ 5*3 + 2] > keypoint1[ 6*3 + 2]:
#             leftFencer_Shoulder = joints_2[5]
#         else:
#             leftFencer_Shoulder = joints_2[6]

#         if keypoint2[ 5*3 + 2] > keypoint2[ 6*3 + 2]:
#             rightFencer_Shoulder = joints_2[5]
#         else:
#             rightFencer_Shoulder = joints_2[6]

#        #find the fencing hand, which is closer t            rightFencer_wrist = rightFencer_RWrist
#             rightFencer_wrist = rightFencer_RWrist
#             rightFencer_wrist = rightFencer_RWristo the other fencer
#         if distance(leftFencer_LWrist, rightFencer_Shoulder) > distance(leftFencer_RWrist, rightFencer_Shoulder):
#             leftFencer_wrist = leftFencer_RWrist
#             leftFencer_elbow = leftFencer_RElbow
#             leftFencer_shoulder = leftFencer_RShoulder
#         else:
#             leftFencer_wrist = leftFencer_LWrist
#             leftFencer_elbow = leftFencer_LElbow
#             leftFencer_shoulder = leftFencer_LShoulder

#         if distance(rightFencer_LWrist, leftFencer_Shoulder) > distance(rightFencer_RWrist, leftFencer_Shoulder):
#             rightFencer_wrist = rightFencer_RWrist
#             rightFencer_wrist = rightFencer_RWrist
#             rightFencer_wrist = rightFencer_RWrist
#         else:
#             rightFencer_wrist = rightFencer_LWrist

#         # check_straightness = 1, complete straight, = 0 the base is the same dis as height
#         # check_straightness can be negative when the base is very small
#         L_is_traightness = check_straightness(leftFencer_wrist, leftFencer_elbow, leftFencer_LShoulder)
#         R_is_traightness = check_straightness(rightFencer_LWrist, rightFencer_LElbow, rightFencer_LShoulder)

#         arm_straight = [False, False]
#         straightness_ratio_threshold = 0.8
#         if L_L_is_traightness >  straightness_ratio_threshold or L_R_is_traightness > straightness_ratio_threshold:
#             arm_straight[0] = True        # if arm_straight == True and hand_in_position == True:
#         #     en_guad_frame_list.append(True)
#         # else:
#         #     if len(en_guad_frame_list) > 15:
#         #         en_guad_frame_dict[key] = lisarm_straightt
#         #     en_guad_frame_list = []
#         if R_L_is_traightness < straightness_ratio_threshold or R_R_is_traightness < straightness_ratio_threshold:
#             arm_straight[1] = True

#         # if arm_straight == True and hand_in_position == True:
#         #     en_guad_frame_list.append(True)
#         # else:en_guad_frame_dict
#         #     if len(en_guad_frame_list) def get_en_guad_frame_list(fencer_dict, frame_dict):
#     en_guad_frame_dict  = defaultdict()
#     fencer_frame_side_dict = defaultdict()
#     for key in fencer_dict[0]:
#         fencer_list = [fencer_dict[0][key], fencer_dict[1][key]]
#         leftFencerExit = False
#         rightFencerExit = False
#         for pose in frame_dict[key]:
#             if pose['idx'] == fencer_dict[0][key]:
#                 fencer_list[0] = pose
#                 leftFencerExit  = True
#             if pose['idx'] == fencer_dict[1][key]:
#                 fencer_list[1] = pose
#                 rightFencerExit = True
#         if leftFencerExit and rightFencerExit:
#             fencer_frame_side_dict[key] = fencer_list

#     en_guad_frame_list = []
#     for key, list in fencer_frame_side_dict.items():

#         keypoint1 = list[0]['keypoints']
#         keypoint2 = list[1]['keypoints']

#         numOfPoint = int(len(keypoint2)/3)
#         joints_1 = []
#         joints_2 = []
#         for i in range(numOfPoint):
#             joint_1 = [keypoint1[i*3],keypoint1[i*3 + 1]]
#             joints_1.append(joint_1)
#             joint_2 = [keypoint2[i*3],keypoint2[i*3 + 1]]
#             joints_2.append(joint_2)

#         # idx_list, minDis = findMinDisKeypoint(list)
#         # if idx_list[0] == 9 or idx_list[0] == 10:
#         #     if  idx_list[1] == 9 or idx_list[1] == 10:
#         #         hand_in_position = True
#         #     else:
#         #         hand_in_position = False

#         # check if arm is straight forward


#         leftFencer_LElbow = joints_1[7]
#         leftFencer_RElbow = joints_1[8]
#         leftFencer_LWrist = joints_1[9]
#         leftFencer_RWrist = joints_1[10]
#         leftFencer_LShoulder = joints_1[5]
#         leftFencer_RShoulder = joints_1[6]
#         rightFencer_LElbow = joints_2[7]
#         rightFencer_RElbow = joints_2[8]
#         rightFencer_LWrist = joints_2[9]
#         rightFencer_RWrist = joints_2[10]
#         rightFencer_LShoulder = joints_2[5]
#         rightFencer_RShoulder = joints_2[6]

#         #find a shoulder as reference for the other fencer
#         if keypoint1[ 5*3 + 2] > keypoint1[ 6*3 + 2]:
#             leftFencer_Shoulder = joints_2[5]
#         else:
#             leftFencer_Shoulder = joints_2[6]

#         if keypoint2[ 5*3 + 2] > keypoint2[ 6*3 + 2]:
#             rightFencer_Shoulder = joints_2[5]
#         else:
#             rightFencer_Shoulder = joints_2[6]

#        #find the fencing hand, which is closer t            rightFencer_wrist = rightFencer_RWrist
#             rightFencer_wrist = rightFencer_RWrist
#             rightFencer_wrist = rightFencer_RWristo the other fencer
#         if distance(leftFencer_LWrist, rightFencer_Shoulder) > distance(leftFencer_RWrist, rightFencer_Shoulder):
#             leftFencer_wrist = leftFencer_RWrist
#             leftFencer_elbow = leftFencer_RElbow
#             leftFencer_shoulder = leftFencer_RShoulder
#         else:
#             leftFencer_wrist = leftFencer_LWrist
#             leftFencer_elbow = leftFencer_LElbow
#             leftFencer_shoulder = leftFencer_LShoulder

#         if distance(rightFencer_LWrist, leftFencer_Shoulder) > distance(rightFencer_RWrist, leftFencer_Shoulder):
#             rightFencer_wrist = rightFencer_RWrist
#             rightFencer_wrist = rightFencer_RWrist
#             rightFencer_wrist = rightFencer_RWrist
#         else:
#             rightFencer_wrist = rightFencer_LWrist

#         # check_straightness = 1, complete straight, = 0 the base is the same dis as height
#         # check_straightness can be negative when the base is very small
#         L_is_traightness = check_straightness(leftFencer_wrist, leftFencer_elbow, leftFencer_LShoulder)
#         R_is_traightness = check_straightness(rightFencer_LWrist, rightFencer_LElbow, rightFencer_LShoulder)

#         arm_straight = [False, False]
#         straightness_ratio_threshold = 0.8
#         if L_L_is_traightness >  straightness_ratio_threshold or L_R_is_traightness > straightness_ratio_threshold:
#             arm_straight[0] = True        # if arm_straight == True and hand_in_position == True:
#         #     en_guad_frame_list.append(True)
#         # else:
#         #     if len(en_guad_frame_list) > 15:
#         #         en_guad_frame_dict[key] = lisarm_straightt
#         #     en_guad_frame_list = []
#         if R_L_is_traightness < straightness_ratio_threshold or R_R_is_traightness < straightness_ratio_threshold:
#             arm_straight[1] = True

#         # if arm_straight == True and hand_in_position == True:
#         #     en_guad_frame_list.append(True)
#         # else:
#         #     if len(en_guad_frame_list) > 15:
#         #         en_guad_frame_dict[key] = lisarm_straightt
#         #     en_guad_frame_list = []
#         #fencer_frame_side_dict[key] = fencer_list
#     return en_guad_frame_dict> 15:
#         #         en_guad_frame_dict[key] = lisarm_straightt
#         #     en_guad_frame_list = []
#         #fencer_frame_side_dict[key] = fencer_list
#     return en_guad_frame_dict


def get_feet_dis_dict(fencer_dict, frame_dict):
    feet_dis_dict = defaultdict()
    fencer_frame_side_dict = defaultdict()
    for key in fencer_dict[0]:
        fencer_list = [fencer_dict[0][key], fencer_dict[1][key]]
        leftFencerExit = False
        rightFencerExit = False
        for pose in frame_dict[key]:
            if pose["idx"] == fencer_dict[0][key]:
                fencer_list[0] = pose
                leftFencerExit = True
            if pose["idx"] == fencer_dict[1][key]:
                fencer_list[1] = pose
                rightFencerExit = True
        if leftFencerExit and rightFencerExit:
            fencer_frame_side_dict[key] = fencer_list

    for key, list in fencer_frame_side_dict.items():
        keypoint1 = list[0]["keypoints"]
        keypoint2 = list[1]["keypoints"]

        numOfPoint = int(len(keypoint2) / 3)
        joints_1 = []
        joints_2 = []
        for i in range(numOfPoint):
            joint_1 = [keypoint1[i * 3], keypoint1[i * 3 + 1]]
            joints_1.append(joint_1)
            joint_2 = [keypoint2[i * 3], keypoint2[i * 3 + 1]]
            joints_2.append(joint_2)

        joints_1[LHip_idx]
        joints_1[RHip_idx]
        joints_1[LKnee_idx]
        joints_1[RKnee_idx]
        leftFencer_LAnkle = joints_1[LAnkle_idx]
        leftFencer_RAnkle = joints_1[RAnkle_idx]
        joints_2[LHip_idx]
        joints_2[RHip_idx]
        joints_2[LKnee_idx]
        joints_2[RKnee_idx]
        rightFencer_LAnkle = joints_2[LAnkle_idx]
        rightFencer_RAnkle = joints_2[RAnkle_idx]

        leftFencer_dis = distance(leftFencer_LAnkle, leftFencer_RAnkle)
        rightFencer_dis = distance(rightFencer_LAnkle, rightFencer_RAnkle)
        disRR = distance(rightFencer_RAnkle, leftFencer_RAnkle)
        disRL = distance(rightFencer_RAnkle, leftFencer_LAnkle)
        disLR = distance(rightFencer_LAnkle, leftFencer_RAnkle)
        disLL = distance(rightFencer_LAnkle, leftFencer_LAnkle)
        inBetween_dis = min(disRR, disRL, disLR, disLL)
        feet_dis_dict[key] = [leftFencer_dis, rightFencer_dis, inBetween_dis]

    return feet_dis_dict


def get_fencer_posture(fencer_dict, frame_dict):
    hand_straightness_dict = get_handstraightness_dict(fencer_dict, frame_dict)
    feet_distance_dict = get_feet_dis_dict(fencer_dict, frame_dict)
    return hand_straightness_dict, feet_distance_dict


def get_handstraightness_dict(fencer_dict, frame_dict):
    hand_straightness_dict = defaultdict()
    fencer_frame_side_dict = defaultdict()
    for key in fencer_dict[0]:
        fencer_list = [fencer_dict[0][key], fencer_dict[1][key]]
        leftFencerExit = False
        rightFencerExit = False
        for pose in frame_dict[key]:
            if pose["idx"] == fencer_dict[0][key]:
                fencer_list[0] = pose
                leftFencerExit = True
            if pose["idx"] == fencer_dict[1][key]:
                fencer_list[1] = pose
                rightFencerExit = True
        if leftFencerExit and rightFencerExit:
            fencer_frame_side_dict[key] = fencer_list

    for key, list in fencer_frame_side_dict.items():
        if key == 190:
            pass
        keypoint1 = list[0]["keypoints"]
        keypoint2 = list[1]["keypoints"]

        numOfPoint = int(len(keypoint2) / 3)
        joints_1 = []
        joints_2 = []
        for i in range(numOfPoint):
            joint_1 = [keypoint1[i * 3], keypoint1[i * 3 + 1]]
            joints_1.append(joint_1)
            joint_2 = [keypoint2[i * 3], keypoint2[i * 3 + 1]]
            joints_2.append(joint_2)

        leftFencer_LElbow = joints_1[LElbow_idx]
        leftFencer_RElbow = joints_1[RElbow_idx]
        leftFencer_LWrist = joints_1[LWrist_idx]
        leftFencer_RWrist = joints_1[RWrist_idx]
        leftFencer_LShoulder = joints_1[LShoulder_idx]
        leftFencer_RShoulder = joints_1[RShoulder_idx]
        rightFencer_LElbow = joints_2[LElbow_idx]
        rightFencer_RElbow = joints_2[RElbow_idx]
        rightFencer_LWrist = joints_2[LWrist_idx]
        rightFencer_RWrist = joints_2[RWrist_idx]
        rightFencer_LShoulder = joints_2[LShoulder_idx]
        rightFencer_RShoulder = joints_2[RShoulder_idx]

        # find a shoulder as reference for the other fencer
        if keypoint1[LShoulder_idx * 3 + 2] > keypoint1[RShoulder_idx * 3 + 2]:
            leftFencer_Shoulder = joints_1[LShoulder_idx]
        else:
            leftFencer_Shoulder = joints_1[RShoulder_idx]

        if keypoint2[LShoulder_idx * 3 + 2] > keypoint2[RShoulder_idx * 3 + 2]:
            rightFencer_Shoulder = joints_2[LShoulder_idx]
        else:
            rightFencer_Shoulder = joints_2[RShoulder_idx]

        # find the
        dis_rightHand = distance(rightFencer_RWrist, leftFencer_Shoulder)
        dis_leftHand = distance(rightFencer_LWrist, leftFencer_Shoulder)
        if dis_rightHand < dis_leftHand:
            rightFencer_wrist = rightFencer_RWrist
            rightFencer_elbow = rightFencer_RElbow
            rightFencer_shoulder = rightFencer_RShoulder
        else:
            rightFencer_wrist = rightFencer_LWrist
            rightFencer_elbow = rightFencer_LElbow
            rightFencer_shoulder = rightFencer_LShoulder

        dis_rightHand = distance(leftFencer_RWrist, rightFencer_Shoulder)
        dis_leftHand = distance(leftFencer_LWrist, rightFencer_Shoulder)
        if dis_rightHand < dis_leftHand:
            leftFencer_wrist = leftFencer_RWrist
            leftFencer_elbow = leftFencer_RElbow
            leftFencer_shoulder = leftFencer_RShoulder
        else:
            leftFencer_wrist = leftFencer_LWrist
            leftFencer_elbow = leftFencer_LElbow
            leftFencer_shoulder = leftFencer_LShoulder

        dis_rightHand = distance(rightFencer_RWrist, leftFencer_Shoulder)
        dis_leftHand = distance(rightFencer_LWrist, leftFencer_Shoulder)
        if dis_rightHand < dis_leftHand:
            rightFencer_wrist = rightFencer_RWrist
            rightFencer_elbow = rightFencer_RElbow
            rightFencer_shoulder = rightFencer_RShoulder
        else:
            rightFencer_wrist = rightFencer_LWrist
            rightFencer_elbow = rightFencer_LElbow
            rightFencer_shoulder = rightFencer_LShoulder

        # check_straightness = 1, complete straight, = 0 the base is the same dis as height
        # check_straightness can be negative when the base is very small
        L_traightness = check_straightness(leftFencer_wrist, leftFencer_elbow, leftFencer_shoulder)
        R_traightness = check_straightness(rightFencer_wrist, rightFencer_elbow, rightFencer_shoulder)
        hand_straightness_dict[key] = [L_traightness, R_traightness]

    return hand_straightness_dict


def main():
    # everything that currently runs at import time
    alphaPose_resuslt_path = "testResults/"
    input_results = os.listdir(alphaPose_resuslt_path)
    # ... the rest of your pipeline ...
    alphaPose_resuslt_path = "testResults/"

    input_results = os.listdir(alphaPose_resuslt_path)
    print(f"The folder {alphaPose_resuslt_path} has {len(input_results)} video files to be processed")
    print(input_results)
    for path in input_results:
        result = alphaPose_resuslt_path + path

        print(result)
        if os.path.isdir(result):
            alphaPose_resuslt_image_path = result + "/vis_orig/"
            alphaPose_resuslt_json_name = result + "/precision_results.json"
            vis_ok = os.path.isdir(alphaPose_resuslt_image_path)
            json_ok = os.path.isfile(alphaPose_resuslt_json_name)

            numberOfImage = len(os.listdir(alphaPose_resuslt_image_path))
            padSize = len(str(numberOfImage))
            if not vis_ok or not json_ok:
                print(f"aphapose result {result} is not ready!")
                continue

            # filtered_video_name = result +"/filtered.avi"
            filtered_json_name = result + "/filtered.json"
            fencer_image_dir = result + "/fencer_image_dir"
            # filtered_image_dir = result+"/filtered_image_dir"
            # S1_remove_dark_image_dir = result+"/s1_image_dir"
            # S2_remove_small_image_dir = result+"/s2_image_dir"

            if os.path.isfile(filtered_json_name):
                print(f"{result} already processed!")
                continue

            if not os.path.isdir(fencer_image_dir):
                os.mkdir(fencer_image_dir)
            # if os.path.isdir(filtered_image_dir) == False:
            #     os.mkdir(filtered_image_dir)
            # if os.path.isdir(S1_remove_dark_image_dir) == False:
            #     os.mkdir(S1_remove_dark_image_dir)

            # print(alphaPose_resuslt_json_name)
            rawText = ""
            with open(alphaPose_resuslt_json_name, encoding="utf-8") as f:
                rawText = f.readline()
                # print(str1[:50])
            str2 = rawText[1:][:-1]
            str3 = str2.split("},{")
            # print(f'start to process {# Video Generating functionresult}, {len(str3)} poses and {numberOfImage} images are in this video.')
            print(f"({alphaPose_resuslt_json_name} file contains {len(str3)} poses")
            # print(str3[:50])
            # json_objs_array = []
            # json_objs_array_orig = []
            # idxes = {}
            # idx_used = 3167
            json_obj_init_list = []

            for i in range(len(str3)):
                if i == 0:
                    jsonStr = str3[i] + "}"
                elif i == len(str3) - 1:
                    jsonStr = "{" + str3[i]
                else:
                    jsonStr = "{" + str3[i] + "}"
                # print("-------")
                # print("i=",i,"    ", jsonStr, "    ++++    ", jsonStr[425:440])
                json_obj = json.loads(jsonStr)
                json_obj_init_list.append(json_obj)
            # print("json_obj_init_list:", json_obj_init_list[:20])
            # image_id, 1st frame, frame No.

            # pose_dict1 = get_pose_dict(json_obj_init_list)
            json_obj_init_list = normaliseIdx(json_obj_init_list)

            json_obj_final_list = []
            json_obj_frame_list = []
            json_obj = json_obj_init_list[0]
            json_obj_frame_list.append(json_obj)
            obj_prev = json_obj
            # frame_fencer_list = [[],[]]
            frame_fencer_dict = [defaultdict(), defaultdict()]
            frame_dict = defaultdict()
            overlap_dict = defaultdict()
            # process pose within a frame, and another frame
            # the loop below uses json_obj_init_list to generate 1) pose_dict, a dictionary containing all the poses
            # grouped by pose idx; 2) a frame_dict, a dictionary containing for the poses grouped by frame idx; and
            # 3) a overlap_list, a list containing pose (box) overlap information. overlap_list is stored inside frame_dict
            # frame_dict = {frame_idx: [pose_list, overlap_list]}
            for i in range(1, len(json_obj_init_list)):
                json_obj = json_obj_init_list[i]
                if json_obj["image_id"] == obj_prev["image_id"] and i != len(json_obj_init_list) - 1:
                    json_obj_frame_list.append(json_obj)
                else:
                    # if i % 1000 == 0:
                    #     print(f'i={i}, img_id={frame_no}, {len(json_obj_frame_list)}')

                    # process the josn_obj of the (prev) frame
                    fileName = obj_prev["image_id"]  # .replaceremove_overlap_fencers(".jpg","")+"_orig.jpg"
                    frame_no = int(fileName.replace(".jpg", ""))
                    if frame_no == 90:
                        pass
                    # print(f'i={i}, img_id={frame_no}, {len(json_obj_frame_list)}')
                    img = cv2.imread(alphaPose_resuslt_image_path + fileName)
                    img = cv2.putText(
                        img, str(frame_no), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                    )

                    out_list_f = removeFlat_2(json_obj_frame_list)

                    out_list_s = removeSmall_2(out_list_f)
                    # outF = f'{S1_remove_dark_image_dir}/img_{str(frame_no).zfill(padSize)}.jpg'
                    # cv2.imwrite(outF, img_out)

                    out_list_b = removeDarkClothing(out_list_s, img)

                    overlap_list = get_Overlap_list(out_list_b)
                    # 2 poses (with different idx) are overlapped in a frame
                    # # if Pose_R != []:
                    # #     right_idx = Pose_R['idx']

                    # # frame_fencer_list.append([left_idx,right_idx])
                    if len(overlap_list) > 0:
                        out_list_b, overlap_list = remove_overlap_by_size(out_list_b, overlap_list, 3)

                    if len(overlap_list) > 0:
                        out_list_b, overlap_list = merge_overlap_pose(out_list_b, overlap_list)

                    if len(overlap_list) > 0:
                        overlap_dict[frame_no] = overlap_list

                    # Pose_L, Pose_R, img= getFencerPose(out_list_b,img)

                    # img_out = drawPose(out_list_b, img)
                    json_obj_final_list = json_obj_final_list + out_list_b  # # if Pose_R != []:
                    # #     right_idx = Pose_R['idx']

                    # # frame_fencer_list.append([left_idx,right_idx])

                    obj_prev = json_obj
                    json_obj_frame_list = []
                    json_obj_frame_list.append(obj_prev)
                    if frame_no % 500 == 0:
                        print(" image frames processed: ", frame_no)
            # end of loop

            filtered_pose_list = merge_poses(json_obj_final_list)  # same person with diff idx in diff frames

            frame_pose_list = []
            list_size = len(filtered_pose_list)
            json_obj = filtered_pose_list[0]
            frame_pose_list.append(json_obj)
            obj_prev = json_obj

            for i in range(1, list_size):
                json_obj = filtered_pose_list[i]
                if json_obj["image_id"] == obj_prev["image_id"] and i != list_size - 1:
                    frame_pose_list.append(json_obj)
                else:
                    fileName = obj_prev["image_id"]  # .replace(".jpg","")+"_orig.jpg"
                    frame_no = int(fileName.replace(".jpg", ""))
                    # if frame_no == 364:
                    #     ttt = 0
                    frame_list = []
                    for pose in frame_pose_list:
                        frame_list.append(pose)

                    obj_prev["image_id"]
                    frame_dict[frame_no] = frame_list

                    Pose_L, Pose_R = getFencerPose(frame_list)
                    # Pose_L, Pose_R, img = getFencerPose(frame_list, img)

                    left_idx = "-1"
                    right_idx = "-1"
                    if Pose_L != [] and Pose_R != []:
                        Box_L = Pose_L["box"]
                        Box_R = Pose_R["box"]
                        if Box_L[0] < Box_R[0] and Box_L[0] + Box_L[2] < Box_R[0] + Box_R[2]:
                            left_idx = Pose_L["idx"]
                            right_idx = Pose_R["idx"]

                    frame_fencer_dict[0][frame_no] = left_idx
                    frame_fencer_dict[1][frame_no] = right_idx

                    obj_prev = json_obj
                    frame_pose_list = []
                    frame_pose_list.append(obj_prev)
                    if frame_no % 500 == 0:
                        print(" image frames processed: ", frame_no)
            # end of loop

            #        print(frame_dict[365])
            frame_fencer_dict = remove_overlap_fencers(frame_fencer_dict, frame_dict)

            # en_guad_dict = get_en_guad_frame_list(frame_fencer_dict, frame_dict)
            # fencer_body_posture [left_fencer, right_fencer, distance_in_between]
            # left_fencer = [hand_straightness, feet_distance, forward_leg_straightness, backward_leg_staightness, truck_tilt, truck_speed]

            hand_straight, _feet_dis = get_fencer_posture(frame_fencer_dict, frame_dict)
            # hand_straight = get_hand_straightness_dict(frame_fencer_dict, frame_dict)
            # feet_dis = get_feet_dis_dict(frame_fencer_dict, frame_dict)
            # draw fencer label
            for key, pose_list in frame_dict.items():
                # for key in range(len(frame_dict)):
                #    pose_list = frame_dict[key]
                # frame_pose = frame_dict[frame]
                fileName = str(key) + ".jpg"  # obj_prev['iget_feet_dis_dictmage_id'] #.replace(".jpg","")+"_orig.jpg"
                img = cv2.imread(alphaPose_resuslt_image_path + fileName)
                cv2.putText(img, str(key), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if key == 190:
                    pass
                if key in hand_straight:
                    for straightness in hand_straight[key]:
                        if straightness > 0.9:
                            cv2.putText(
                                img,
                                "straight Arm",
                                (590, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                5,
                                (0, 0, 255),
                                5,
                                cv2.LINE_AA,
                            )

                drawPoseBBox(pose_list, img)
                drawPose(pose_list, img)

                # Pose_L, Pose_R = getFencerPose(pose_list)
                Pose_idx_L = frame_fencer_dict[0][key]
                Pose_idx_R = frame_fencer_dict[1][key]

                if Pose_idx_L != "-1":
                    for pose in pose_list:
                        if pose["idx"] == Pose_idx_L:
                            box = pose["box"]
                            p = (int(box[0]), int(box[1] + box[3]))
                            cv2.putText(img, "L", p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if Pose_idx_R != "-1":
                    for pose in pose_list:
                        if pose["idx"] == Pose_idx_R:
                            box = pose["box"]
                            p = (int(box[0]), int(box[1] + box[3]))
                            cv2.putText(img, "R", p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # frame_no = key
                # if frame_no == 90:
                #     t = 0
                # print(f'i={i}, img_id={frame_no}, {len(json_obj_frame_list)}')

                outF = f"{fencer_image_dir}/img_{str(key).zfill(padSize)}.jpg"
                cv2.imwrite(outF, img)

            f = open(filtered_json_name, "w")
            json_string = json.dumps(json_obj_final_list)
            f.writelines(json_string)
            f.close()


if __name__ == "__main__":
    main()
