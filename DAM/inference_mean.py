# an inference code about pretrained_model(DAM_model.pth)
# model training code : train_gaze360.py
# pretrained model's detail during training is written in gaze360.log

# edit demo_2.py code
import os
import glob
import torch
import pandas as pd
import numpy as np
from model_demo import GazeTargetDetectionNet
from PIL import Image
from torchvision import transforms
import cv2
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GazeTargetDetectionNet().to(device)
model_state_dict = torch.load('./DAM_model_epoch50.pth', map_location=device)
model.load_state_dict(model_state_dict, strict=False)
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--annotation_dir', type=str, help='from gaze360 csv file',
                    default='/home/soo/gaze_total/gaze/gaze360/output/csv/output/')
parser.add_argument('--data_dir', type=str, help='image folder path',
                    default='/home/soo/gaze_total/gaze/DAM/data/videoattentiontarget/images/')
parser.add_argument('--depth_dir', type=str, help='depthmap folder path',
                    default='/home/soo/gaze_total/gaze/MiDaS/output_videoattentiontarget/')
args = parser.parse_args()
WIDTH = 1152
HEIGHT = 720
fps = 15
###############################################################################

transform_head = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_depth = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# FOV generator


def generate_data_field(head_position):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape(
        [1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape(
        [height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = head_position
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid


def get_info(dataframe, data_path, depthmap_path):
    path, head_x_min, head_y_min, head_x_max, head_y_max, gaze_x, gaze_y, gaze_z = dataframe
    head_x_min = head_x_min[1:-1].split(',')
    head_y_min = head_y_min[1:-1].split(',')
    head_x_max = head_x_max[1:-1].split(',')
    head_y_max = head_y_max[1:-1].split(',')
    gaze_x = gaze_x[1:-1].split(',')
    gaze_y = gaze_y[1:-1].split(',')
    gaze_z = gaze_z[1:-1].split(',')
    head_l = []
    gaze_field_l = []
    normalized_direction_l = []

    head_channel_l = []
    head_xy_l = []
    gaze_z_l = []
    for person in range(0, len(head_x_min)):
        head_channel = torch.zeros(224, 224)
        gaze_x_person, gaze_y_person, gaze_z_person = gaze_x[person], gaze_y[person], gaze_z[person]

        gaze_x_person = float(gaze_x_person)
        gaze_y_person = float(gaze_y_person)
        gaze_z_person = float(gaze_z_person)
        gaze_x_person = torch.as_tensor(gaze_x_person)
        gaze_y_person = torch.as_tensor(gaze_y_person)
        gaze_z_person = torch.as_tensor(gaze_z_person)
        gaze_z_person = torch.mul(gaze_z_person, -1)
        gaze_z_person = gaze_z_person.view(-1, 1, 1)
        gaze_x_person = torch.mul(gaze_x_person, -1).view(-1, 1)
        gaze_y_person = torch.mul(gaze_y_person, -1).view(-1, 1)

        normalized_direction = torch.stack((gaze_x_person, gaze_y_person))

        img = Image.open(os.path.join(data_path, path))
        image_num_str = path.split('.')[0]
        depth_path = image_num_str + '-dpt_beit_large_512.png'
        img = img.convert('RGB')

        width, height = img.size
        imsize = torch.IntTensor([width, height])

        # RGB -> gray scale image  (channel 부분의 차원도 변화)
        depthmap = Image.open(os.path.join(
            depthmap_path, depth_path)).convert("L")

        depthmap = transform_depth(depthmap).squeeze(0)

        head_x_min__ = int(head_x_min[person])
        head_x_max__ = int(head_x_max[person])
        head_y_min__ = int(head_y_min[person])
        head_y_max__ = int(head_y_max[person])

        head_x_min__ = head_x_min__ / WIDTH * width
        head_x_max__ = head_x_max__ / WIDTH * width
        head_y_min__ = head_y_min__ / HEIGHT * height
        head_y_max__ = head_y_max__ / HEIGHT * height

        head_x_min__ = int(head_x_min__)
        head_x_max__ = int(head_x_max__)
        head_y_min__ = int(head_y_min__)
        head_y_max__ = int(head_y_max__)

        w = abs(head_x_max__ - head_x_min__)
        h = abs(head_y_max__ - head_y_min__)
        head_x_min__ = int(head_x_min__ - 0.05 * w)
        head_y_min__ = int(head_y_min__ - 0.1 * h)
        head_y_max__ = int(head_y_max__ - 0.3 * h)
        head = img.crop((head_x_min__, head_y_min__,
                        head_x_max__, head_y_max__))
        head = transform_head(head)
        image = transform_img(img)

        head_position = [(head_x_min__+head_x_max__)/(2*width),
                         (head_y_min__+head_y_max__)/(2*height)]
        gaze_field = generate_data_field(head_position)
        gaze_field = torch.FloatTensor(gaze_field).to(device)

        head_channel[int(head_position[1]*224), int(head_position[0]*224)] = 1
        head_position = torch.FloatTensor(head_position).to(device)
        head_l.append(head)
        gaze_field_l.append(gaze_field)
        head_channel_l.append(head_channel)  # change!!
        head_xy_l.append([head_x_min__, head_x_max__,
                         head_y_min__, head_y_max__])
        normalized_direction_l.append(normalized_direction)
        gaze_z_l.append(gaze_z_person)
    return image, depthmap, head_l, head_channel_l, gaze_field_l, imsize, head_xy_l, normalized_direction_l, gaze_z_l


if __name__ == '__main__':
    input_depthmap = torch.rand(4, 224, 224)
    input_head = torch.rand((4, 3, 224, 224))
    input_head_channel = torch.zeros(4, 224, 224)
    input_head_channel[:, 1, 1] = 1
    input_gaze_field = torch.rand((4, 2, 224, 224))

    all_annotation_dir_paths = sorted(
        glob.glob(os.path.join(args.annotation_dir, '*',  '*.csv')))  # 606개
    all_data_dir_paths = sorted(
        glob.glob(os.path.join(args.data_dir, '*', '*')))
    all_depth_dir_paths = sorted(
        glob.glob(os.path.join(args.depth_dir, '*', '*')))

    cnt = 0

    for path in range(0, len(all_annotation_dir_paths)):
        print('path: ', path)
        df = pd.read_csv(
            all_annotation_dir_paths[path], sep=',', index_col=False)
        df = df[['path', 'head_x_min', 'head_y_min', 'head_x_max',
                 'head_y_max', 'gaze_x', 'gaze_y', 'gaze_z']]
        data_path = all_data_dir_paths[path]
        depthmap_path = all_depth_dir_paths[path]
        frame_path = sorted(glob.glob(os.path.join(data_path, '*.jpg')))

        frame_cnt = len(frame_path)
        model.eval()
        with torch.no_grad():
            W = max(int(fps//5), 1)
            for i in range(0, frame_cnt):
                print('frame : ', i)
                image, depthmap, head_l, head_channel_l, gaze_field_l, imsize, head_xy_l, normalized_direction_l, z_l = get_info(
                    df.iloc[i], data_path, depthmap_path)
                #print('head 의 수 : ', len(head_l))
                frame = cv2.imread(frame_path[i])
                for person in range(0, len(head_l)):
                    input_image = torch.zeros(4, 3, 224, 224)
                    count = 0
                    for j in range(i-2*W, i+2*W, W):
                        input_image[count, :, :, :] = image
                        input_depthmap[count, :, :] = depthmap[0]
                        input_head[count, :, :, :] = head_l[person]
                        # change!!
                        input_head_channel[count, :,
                                           :] = head_channel_l[person]
                        input_gaze_field[count, :, :, :] = gaze_field_l[person]
                        count = count + 1

                    x = input_image.cuda(), input_depthmap.cuda(), input_head.cuda(), input_head_channel.cuda(
                    ), input_gaze_field.cuda(), normalized_direction_l[person].cuda(), z_l[person].cuda()

                    normalized_direction, output, inout, attention_map = model(
                        x)

                    head_x_min__, head_x_max__, head_y_min__, head_y_max__ = head_xy_l[person]
                    head_center = (int((head_x_min__ + head_x_max__)/2.0),
                                   int((head_y_min__ + head_y_max__)/2.0))

                    norm_x = normalized_direction[0].item()
                    norm_y = normalized_direction[1].item()

                    des = (int(head_center[0] - norm_x*30),
                           int(head_center[1] - norm_y*30))

                    img_box = cv2.rectangle(
                        frame, (head_x_min__, head_y_min__), (head_x_max__, head_y_max__), (255, 0, 0), 3)

                    output = output[0].cpu().detach().numpy() * 255
                    output = output.squeeze()
                    inout = inout[0].cpu().detach().numpy()
                    inout = 1 / (1 + np.exp(-inout))
                    inout = (1 - inout) * 255
                    from scipy.misc import imresize
                    import matplotlib.pyplot as plt

                    norm_map = imresize(output, (imsize[1], imsize[0])) - inout

                    norm_map += frame[:, :, 0]
                    plt.imshow(norm_map, cmap='jet',
                               alpha=0.2, vmin=0, vmax=255)
                    plt.savefig('./ddd/'+str(cnt))

                cnt = cnt + 1
