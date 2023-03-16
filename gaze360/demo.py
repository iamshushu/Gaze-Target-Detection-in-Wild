import pandas as pd
from pandas import DataFrame
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import imageio
import cv2
import random
import argparse
import glob
from PIL import Image

from model import GazeLSTM, PinBallLoss

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
setup_logger()

######################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--source_path', type=str, help='input(image) folder path', default='/home/soo/gaze_total/gaze/DAM/data/videoattentiontarget/images/')
parser.add_argument('--output_csv_save_path', type=str, help='output(csv) folder path', default='/home/soo/gaze_total/gaze/gaze360/output/csv/output/')
args = parser.parse_args()
WIDTH, HEIGHT = 1152, 720
fps = 15
######################################################################

def get_detectron2_predictor():
    # Use Detectron2 to find people body bounding boxes (x0, y0, x1, y1)
    # create a detectron2 config & a detectron2 DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


def get_head_bbox(body_bbox):
    # body bbox -> head bbox
    # TODO: Improve head location
    x0 = body_bbox[0]
    x1 = body_bbox[2]
    y0 = body_bbox[1]
    y1 = body_bbox[3]
    w = abs(x1-x0)
    h = abs(y1-y0)
    # y 축은 위로 갈수록 값이 작아짐
    # x0, y0 (왼쪽 위 좌표) < x1, y1 (오른쪽 아래 좌표)
    if w <= (h * 0.5):
        return np.array((x0 + w*0.1, y0 + h*0.1, x1 - w*0.1, y1 - h*0.6), dtype=np.float32)

    return np.array((x0 + w*0.1, y0 + h*0.1, x1 - w*0.1, y1 - h*0.3), dtype=np.float32)


def extract_heads_bbox(body_bbox):
    N = len(body_bbox)
    if N == 0:
        return []
    bbox_list = []
    for i in range(N-1, -1, -1):
        bbox = get_head_bbox(body_bbox[i])
        if bbox is None:
            continue
        bbox_list.append(bbox)
    bbox_list = np.array(bbox_list, float)
    return bbox_list


def compute_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    bb2_area = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    eps = 1e-8

    if iou <= 0.0 or iou > 1.0 + eps:
        return 0.0

    return iou


def find_id(bbox, id_dict):
    id_final = None
    max_iou = 0.5
    for k in id_dict.keys():
        if(compute_iou(bbox, id_dict[k][0]) > max_iou):
            id_final = k
            max_iou = compute_iou(bbox, id_dict[k][0])
    return id_final


def spherical2cartesial(x):
    output = torch.zeros(x.size(0), 3)
    output[:, 2] = -torch.cos(x[:, 1])*torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1])*torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])
    return output


def denormalize(image):
    return image.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])


def main():
    data = ['path', 'head_x_min', 'head_y_min', 'head_x_max',
            'head_y_max', 'gaze_x', 'gaze_y', 'gaze_z']
    data_value = []
    jpg_path = []
    for (root, directories, files) in os.walk(args.source_path):
        for file in files:
            if '.jpg' in file:
                jpg_path.append(os.path.join(root, file))
    video_clip_list = [] 
    for path in jpg_path:
        split_path = path.split('/')
        video, clip, frame = split_path[-3:]
        video_clip_list.append((video, clip))
    video_clip_list = list(set(video_clip_list))
    
    # 3D Gaze Direction Prediction Model
    model_v = GazeLSTM()
    model = torch.nn.DataParallel(model_v).cuda()
    model.cuda()
    checkpoint = torch.load('./gaze360_model.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    for video, clip in video_clip_list:
        img_path_list = glob.glob(os.path.join(args.source_path + video + '/' + clip + '/', '*.jpg'))
        img_path_list.sort()
        image_list = []
        for i in range(0, len(img_path_list)):
            image = cv2.imread(img_path_list[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list.append(image)
        v = list(image_list)
        predictor = get_detectron2_predictor()
        outputs = []
        for i in range(0, len(img_path_list)):
            outputs.append(predictor(image_list[i]))
        bbox_result = []
        for i in range(0, len(outputs)):
            body_bbox = outputs[i]['instances'][outputs[i]['instances'].pred_classes==0].pred_boxes.tensor.cpu().numpy()
            bbox_result.append(extract_heads_bbox(body_bbox))
            names = []
            for i in range(0, len(bbox_result)):
                names.append(i)
            names = np.array(names)
            final_results = {n: b for n, b in zip(
                names, bbox_result) if len(bbox_result) > 0}
        id_num = 0
        tracking_id = dict()
        identity_last = dict()
        frames_with_people = list(final_results.keys())
        frames_with_people.sort()

        for i in frames_with_people:
            speople = final_results[i]
            identity_next = dict()
            for j in range(len(speople)):
                bbox_head = speople[j]
                if bbox_head is None: continue
                id_val = find_id(bbox_head, identity_last)
                if id_val is None:
                    id_num += 1
                    id_val = id_num
                # TODO: Improve eye location
                eyes = [(bbox_head[0]+bbox_head[2])/2.0,(0.6*bbox_head[1]+0.4*bbox_head[3])]
                identity_next[id_val] = (bbox_head, eyes)
            identity_last = identity_next
            tracking_id[i] = identity_last

        color_encoding = []
        for i in range(1000):
            color_encoding.append(
                [random.randint(0, 254), random.randint(0, 254), random.randint(0, 254)])
        image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        W = max(int(fps//8), 1)
        data_value = []
        for i in range(0, len(v)):
            head_x_min = [] 
            head_y_min = [] 
            head_x_max = [] 
            head_y_max = []
            gaze_x = [] 
            gaze_y = [] 
            gaze_z = []
            image = v[i].copy()
            image = cv2.resize(image, (WIDTH, HEIGHT))
            image = image.astype(float)
            with torch.no_grad():
                for id_t in tracking_id[i].keys():
                    input_image = torch.zeros(7, 3, 224, 224)
                    # 연속된 7개의 frame 모두 확인
                    count = 0
                    for j in range(i-3*W, i+4*W, W):
                        if j in tracking_id and id_t in tracking_id[j]:
                            new_im = Image.fromarray(v[j], 'RGB')
                            bbox, eyes = tracking_id[j][id_t]
                        else:
                            new_im = Image.fromarray(v[i], 'RGB')
                            bbox, eyes = tracking_id[i][id_t]
                        new_im = new_im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        input_image[count, :, :, :] = image_normalize(
                            transforms.ToTensor()(transforms.Resize((224, 224))(new_im)))
                        count = count+1

                    bbox, eyes = tracking_id[i][id_t]
                    bbox = np.asarray(bbox).astype(int)
                    bbox[0], bbox[2] = WIDTH*bbox[0] / \
                        v[i].shape[1], WIDTH*bbox[2]/v[i].shape[1]
                    bbox[1], bbox[3] = HEIGHT*bbox[1] / \
                        v[i].shape[0], HEIGHT*bbox[3]/v[i].shape[0]
                    eyes = np.asarray(eyes).astype(float)
                    eyes[0], eyes[1] = WIDTH * eyes[0] / \
                        float(v[i].shape[1]), HEIGHT * eyes[1]/float(v[i].shape[0])
                    head_center_x = eyes[0]
                    head_center_y = eyes[1]
                    head_center = (int(head_center_x), int(head_center_y))

                    output_gaze, _ = model(input_image.view(1, 7, 3, 224, 224).cuda())
                    #print('output gaze : ', output_gaze)
                    gaze = spherical2cartesial(output_gaze).detach()
                    #print('gaze (x,y,z) = cartesial gaze: ', gaze)
                    gaze_dir_2d = gaze[0, 0:2].numpy()
                    # print('gaze_dir_2d : ', gaze_dir_2d)
                    # 3D -> 2D
                    gaze_dir_2d /= np.linalg.norm(gaze_dir_2d)

                    des = (int(head_center[0] - gaze_dir_2d[0]*30),
                        int(head_center[1] - gaze_dir_2d[1]*30))
                    img_arrow = cv2.arrowedLine(
                        image.copy(), head_center, des, (0, 255, 0), 4, tipLength=0.8)
                    binary_img = (
                        (img_arrow[:, :, 0]+img_arrow[:, :, 1]+img_arrow[:, :, 2]) == 0.0).astype(float)
                    binary_img = np.reshape(binary_img, (HEIGHT, WIDTH, 1))
                    binary_img = np.concatenate(
                        (binary_img, binary_img, binary_img), axis=2)

                    image = binary_img*image + img_arrow*(1-binary_img)

                    image = image.astype(np.uint8)

                    image = cv2.rectangle(
                        image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_encoding[min(id_t, 900)], 2)
                    image = image.astype(float)
                    head_x_min.append(bbox[0])
                    head_y_min.append(bbox[1])
                    head_x_max.append(bbox[2])
                    head_y_max.append(bbox[3])
                    gaze_x.append(gaze[0][0].item())
                    gaze_y.append(gaze[0][1].item())
                    gaze_z.append(gaze[0][2].item())
                    # 화살표만 출력
                    #image = img_arrow
                # 이미지 내 마지막 사람의 정보 저장
                video_folder = (img_path_list[i].split('/'))[-3]
                clip_folder = (img_path_list[i].split('/'))[-2]
                jpg_name = (img_path_list[i].split('/'))[-1]
                
                path = video_folder + '/' + clip_folder + '/' + jpg_name
                data_value.append([jpg_name, head_x_min, head_y_min,    
                                head_x_max, head_y_max, gaze_x, gaze_y, gaze_z])
                image = image.astype(np.uint8)
                #out.append_data(image)
            print('path : ', path)
            df = pd.DataFrame(data_value, columns=data)

            if not os.path.exists(args.output_csv_save_path + video_folder):
                os.makedirs(args.output_csv_save_path + video_folder)
            df.to_csv(args.output_csv_save_path + video_folder + '/' + clip_folder + '.csv', sep=',', na_rep='NaN')

if __name__ == '__main__':
    main()
