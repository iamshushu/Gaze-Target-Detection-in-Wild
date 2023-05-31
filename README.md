# Gaze-Target-Detection-in-Wild

<aside>

💻 **Three-stage method to simulate the human gaze inference behavior in 3D space**

    Stage 1) Estimate a 3D gaze orientation from the head

    Stage 2) Develop a Dual Attention Module (DAM)

    Stage 3) Use the generated dual attention as guidance to perform two sub-tasks: 

    Stage 3-1) Identifying whether the gaze target is inside or out of the image 

    Stage 3-2) Locating the target if inside

    
    
    
</aside>

> **논문에서 발전시킨 부분 !!**
> 


**💡 Gaze target detection aims to infer where people in a scene are looking**


**💡 Person in scene is looking (→ people in a scene are looking : 최종 목표 )**


**💡 The Common Gaze Point of Human Observer**


# Manual (Following model 1️⃣ → 2️⃣ → 3️⃣)


## 1️⃣ **Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer**

> MiDaS github link
> 

[https://github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS)

### MiDaS Manual

### <Setup>

1. Terminal Command (Clone a remote Git repository locally)
    
    ```powershell
    git clone https://github.com/isl-org/MiDaS.git
    ```
    
2. Pick one or more models and download the corresponding weights to the `weights` folder
    
    ```
    Download model : [dpt_beit_large_512](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt) 
    -> For highest quality
    ```
    
3. Set up dependencies :
    - Use environment.yaml
        
        ([https://github.com/islorg/MiDaS/blob/master/environment.yaml](https://github.com/isl-org/MiDaS/blob/master/environment.yaml))
        
        ```bash
        # environment.yaml
        name: midas-py310
        channels:
          - pytorch
          - defaults
        dependencies:
          - nvidia::cudatoolkit=11.7
          - python=3.10.8
          - pytorch::pytorch=1.13.0
          - torchvision=0.14.0
          - pip=22.3.1
          - numpy=1.23.4
          - pip:
            - opencv-python==4.6.0.66
            - imutils==0.5.4
            - timm==0.6.12
            - einops==0.6.0
        ```
        
    
    ```
    conda env create -f environment.yaml
    conda activate midas-py310
    ```
    

### <Usage>

1. Place one or more input images in the folder `input`.
    
    (+ input folder도 넣을 수 있음)
    
    ```
    **Input Dataset 1: VideoAttentionTarget**
    Download the VideoAttentionTarget dataset from https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0
    
    **~~Input Dataset 2: GazeFollow~~**
    
    (Source : https://github.com/ejcgt/attention-target-detection)
    ```
    
2. Run the model with
    
    ```python
    python run.py --model_type <model_type> --input_path input --output_path output
    ```
    
    where `<model_type>` is chosen from [dpt_beit_large_512](https://github.com/isl-org/MiDaS#model_type)
    
3. The resulting depth maps are written to the `output` folder.
    
    → Get depth map images  ~~and pfm file~~ 
    

## 2️⃣ Gaze360: Physically Unconstrained Gaze Estimation in the Wild Dataset

> Gaze360 github link
> 

[https://github.com/erkil1452/gaze360](https://github.com/erkil1452/gaze360)

- **Details ( Data, Structure )**
    - **Data**
        
        ```
        Gaze360 Dataset more information at http://gaze360.csail.mit.edu.
        1. Collected in indoor and outdoor environments; 
        	 A wide range of head poses and distances between subjects and cameras.
        2. This repository provides already processed txt files with the split for training the Gaze360 model. The txt contains the following information:
        		Row 1: Image path
        		Row 2-4: Gaze vector
        ```
        
    
    - **Structure (Original)**
        
        ```bash
        ${PROJECT}
        ├── code
        │	├── data_loader.py
        │	├── model.py
        │	├── resnet.py
        │	├── run.py
        │	├── test.txt
        │	├── train.txt
        │	└── validation.txt
        └── dataset
        ```
        
    
    - Structure (Edit)
        
        ```bash
        ${PROJECT}
        ├── detectron2 (폴더)
        ├── output
        │	└── *.csv
        ├── demo.py
        ├── frame_to_video.py
        ├── gaze360_model.pth.tar (pretrained model)
        ├── gaze360.py
        ├── model.py
        └── resnet.py
        ```
        
- 기존 code 에서 변경된 부분
    
    → pretrained 된 model을 사용하므로, 불필요한 파일은 사용 x
    
    → output/csv 부분은 demo.py에서 나온 output(.csv)을 저장하는 folder
    
    → [demo.py](http://demo.py) 는 아래 <Usage> 부분에서 설명함, 실제로 사용하는 부분
    
    → frame_to_video.py는 frame을 video로 만드는 경우에 사용, output을 video로 보고 싶을 때 사용
    
    →gaze360.py 는 아래 <Setup> 부분에서 설명함, 모델을 test 할 수 있는 부분 (참고용)
    
    → 나머지는 동일 ..
    

### Gaze360 Manual

### <Setup>

- Requirements
    
    ```bash
    PyTorch 1.1.0 (but it is likely to work on previous version of PyTorch as well.)
    ```
    
1. Terminal Command (Clone a remote Git repository locally)
    
    ```powershell
    git clone https://github.com/erkil1452/gaze360.git
    ```
    
2. Installation
    
    ```bash
    # 가상환경 생성
    conda create -n gaze360-py37 python=3.7
    conda activate gaze360-py37
    
    # Install DensePose as package
    pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
    
    ######################
    # Fix lot of imports #
    ######################
    pip install imgaug==0.2.6
    pip install opencv-python==4.1.2.30
    # lucid 이용하기 위해 tensorflow 설치 
    # (lucid is not currently supporting tensorflow 2)
    pip install tensorflow==1.15
    pip install tensorflow-gpu==1.15
    
    pip install lucid==0.2.3
    pip install moviepy
    
    # Download Detectron2 source.
    git clone https://github.com/facebookresearch/detectron2.git
    
    # Check whether the DensePose installation was successful:
    # !python detectron2/tests/test_model_analysis.py
    ```
    
3. Download Trained Model Weights from 
    
    [](https://gaze360.csail.mit.edu/files/gaze360_model.pth.tar)
    
4. Download video(video.mp4) ~~and run DensePose on it~~    
    
    (이 4번 부분은 Gaze360 모델을 가장 **간단하게 test**할 수 있는 방법을 설명, 
    
    **실제 사용 예시는 아래의 <Useage>** 보기!)
    
    > Extract the frames of the video in a temporary folder:
    > 
    1. Make the folder 
        
        `mkdir content`
        
        `cd content`
        
        `mkdir temp`
        
        `cd ..`
        
        → 최종 : ‘root directory/content/temp’  folder 생성
        
        →현재 directiory : root directory 
        
    2. Download the video
        
        `wget http://gaze360.csail.mit.edu/files/video.mp4`
        
    3. Extract the frames  (output frames path : ‘root directory/content/temp’ )
        
        `ffmpeg -i video.mp4 ./content/temp/frame%04d.jpg` 
        
    4. test code 명 : **gaze360.py**
        
        ```python
        python gaze360.py
        ```
       
    ---
    
    - Run DensePose Part (사용x, 참고용)
        1. Detect the pose of all the people in the video
            
            `rm -r ./content/DensePose/DensePoseData/infer_out/*`
            
        2. Download model
            
            ```bash
            cd detectron2/projects/DensePose && wget [https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl](https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl)
            ```
            
        3. Test model
            
            ```bash
            #cd detectron2/projects/DensePose
            python apply_net.py show configs/densepose_rcnn_R_101_FPN_s1x.yaml model_final_c6ab63.pkl /content/temp/frame0001.jpg dp_contour,bbox --output image_densepose_contour.png
            ```
            
        4. Run DensePose on the video frames
            1.  **root directory** 
                
                → ****사용자에 따라 달라짐, 변경 필요**
                
            
            ```bash
            mkdir DensePoseData
            #cd detectron2/projects/DensePose 
            python apply_net.py \
              dump \
              configs/densepose_rcnn_R_101_FPN_s1x.yaml model_final_c6ab63.pkl \
              **<root directory>**/content/temp \
              -v \
              --output <root directory>**/content/DensePoseData/data.pkl
            ```
            
        5. Run code
            
            그 전에 현재 directory 확인 !
            
            root directory에서 demo code 실행
            
     ---   

    
    
### Usage

1. Run [demo.py](http://demo.py)
    
    사용자에 따라 다른 path를 사용하므로 변경 필요
    
    ```
    python demo.py --source_path <source_path> --output_csv_save_path <output_csv_save_path>
    ```
    
    - source_path : input(image) 이미지가 들어있는 folder path, 위의 MiDaS input folder와 동일
    - output_csv_save_path : output(csv)을 저장할 folder path
    
    - `input type` : image folder
        - VideoAttentionTarget의 경우, image folder 의 구성
            
            ex) ../data/videoattentiontarget/images/All in the Family/1023_1249/00001029.jpg
            
        - input 과 관련된 변수 설명
            - img_path_list
                
                : image folder 내에 있는 .jpg 파일들의 path를 저장한 list
                
            - image_list
                
                : img_path_list 내 .jpg 파일을 읽은 걸 저장한 list
                
    
    - `output type` : csv 파일
        - VideoAttentionTarget의 경우, output csv 형태
            
            ex) ../output/csv/output/All in the Family/1023_1249.csv
            
        - output csv 내용
            
            ```
            # frame 내의 사람들의 정보 저장
            # 같은 frame 내에 있는 사람이면 path 는 동일
            # frame 내에 여러 사람이 있는 경우, head_x_min, head_y_min,head_x_max, head_y_max, gaze_x, gaze_y, gaze_z 는 모두 list 형태로 저장
            [path, head_x_min, head_y_min,head_x_max, head_y_max, 
            gaze_x, gaze_y, gaze_z]
            ```
            

### 기존 Model에서 달라진 부분 설명

1. DensePose 대신 <Detectron2>의 object detection 사용
    
    `input type` : image 
    
    `output type` : image 내 사람들의 bounding Boxes 좌표 (x1, y1, x2, y2)
    
    이때, 사람의 body bounding box 를 기준으로 사람의 head bounding box 좌표 추출
    
    → 이 부분은 개선 필요 !! 임의의 비율대로 자름
    
    → 얼굴 영역을 잘 잡아내지 못하는 경우 추정하는 3D gaze direction이 정확하지 않을 수 있음
    
    → Detectron2 를 사용하지 않더라도, 사람의 얼굴 영역의 bounding box 만 잘 추출할 수 있다면,   다른 모델 사용 가능!!
    
2. 추출한 사람의 head bounding box 좌표로 crop한 head 이미지를 Gaze360의 pretrained model에 대입후 3D gaze direction을 추정

## 3️⃣ Dual Attention Guided Gaze Target Detection in the Wild

> DAM github link
> 

[https://github.com/Crystal2333/DAM](https://github.com/Crystal2333/DAM)

- Details ( Structure ) & Code
    
    
    - Structure ( Original )
        
        ```bash
        ${PROJECT}
        ├── dataset.py
        ├── gaze_model.py
        ├── model.py
        ├── resnet_scene.py
        ├── train_gazefollow.py
        ├── train_videoatttarget.py
        └── util.py
        ```
        
    
    - Structure ( Edit )
        
        ```bash
        ${PROJECT}
        ├── data
        ├──└── videoattentiontarget
        ├──  ├── annotations
        ├──  └── images
        ├── DAM_model_epoch50.pth
        ├── dataset.py
        ├── gaze_model.py
        ├── inference_edit.py
        ├── inference_mean.py
        ├── model_demo.py
        ├── model_train.py
        ├── model.py
        ├── resnet_scene.py
        ├── train_DAM_model_epoch50.py
        └── util.py
        ```
        
    - 기존 code 에서 변경된 부분 (# [ Original ] 과 # [ Edit ] 부분 )
        - data 는 videoattentiontarget data
            
            → train_DAM_model_epoch50.py 에서 model을 train 할 때 사용하는 input data
            
            - annotation
                
                : frame 이름, 그 frame 내 사람들의 head bounding box 좌표와 gaze target 의 좌표가 .txt 형태로 제공  (ex) s00.txt s01.txt)
                
                : frame.jpg head_bbox_x1, head_bbox_y1, head_bbox_x2, head_bbox_y2, target_x, target_y
                
                : gaze target 이 없는 경우, target_x, target_y 는 각각 -1, -1 로 제공
                
            - images
        - DAM_model_epoch50.pth 는 train_DAM_model_epoch50.py 에서 train 한 모델을 저장한 것
        
### DAM Manual

### <Setup>

위 모델들의 setup 환경과 동일

> 사용할 Dataset
> 

Download the VideoAttentionTarget dataset from 

[https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0)

```
[VideoAttentionTarget dataset]
1. Image
- 1,331 video clips collected from various sources on YouTube
**********2. Annotation**********
- 164,541 frame-level head bounding boxes
- 109,574 in-frame gaze targets
- 54,967 out-of-frame gaze indicators
```

- 이후에 추가로 사용할 수 있는 Datasets : **GazeFollow dataset, Gaze360**

### Usage

1. Pretrained model이 없으므로 train 필요
    - train에 사용할 dataset : VideoAttentionTarget
    - train할 때 사용할 code : train_videoatttarget.py
    
    *사용자에 따라 다른 path를 사용하므로 변경 필요
    
    ```python
    data_dir = '/data/videoattentiontarget/images'
    depth_dir = '/data/videoattentiontarget/depthmap'
    train_annotation_dir = '/data/videoattentiontarget_new/annotations/train'
    test_annotation_dir = '/data/videoattentiontarget_new/annotations/test'
    ```
    
    > data_dir : VideoAttentionTarget dataset 중 image의 path
    > 
    
    > depth_dir : VideoAttention Target image를 이용하여 구한 depthmap의 path
    * 위의 MiDaS를 이용하여 구한 depthmap이므로 MiDaS output path와 동일해야 함 !!
    > 
    
    > train_annotation_dir : VideoAttentionTarget dataset 중 train annotation의 path
    > 
    
    > test_annotation_dir : VideoAttentionTarget dataset 중 test annotation의 path
    > 
    
    - output
        
        root folder 내에 model 저장
        
         DAM_model.pth
        

2-1 . Run inference_edit.py

- trained model 이용
    
    * 사용자에 따라 다른 path를 사용하므로 변경 필요
    
    ```
    python inference_edit.py --annotation_dir <annotation_dir> --data_dir <data_dir> --depth_dir <depth_dir>
    ```
    
- `input type`
    - annotation_dir : gaze360 에서 구한 csv file이 저장된 path
    - data_dir : image folder의 path (MiDas gaze360에서 사용한 input image의 path와 동일)
    - depth_dir : MiDaS에서 구한 depth map image가 저장된 path
- `output type`
    - **예측된 gaze target을 visualize한 image  → backbone 이전의 dual attention image**
    

2-2 . Run inference_mean.py

- trained model 이용
    
    *사용자에 따라 다른 path를 사용하므로 변경 필요
    
    ```
    python inference_mean.py --annotation_dir <annotation_dir> --data_dir <data_dir> --depth_dir <depth_dir>
    ```
    
- `input type`
    - annotation_dir : **gaze360** 에서 구한 csv file이 저장된 path
    - data_dir : image folder의 path (MiDas, gaze360에서 사용한 input image의 path와 동일)
    - depth_dir : **MiDaS**에서 구한 depth map image가 저장된 path
- `output type`
    - 예측된 gaze target을 visualize한 image→ backbone 이후의 Output heatmap
