from statistics import mean
import cv2
import numpy as np
import os

data_root_train = '/root/mv_ca/dataset_full/clips_v1.5'
data_root_val = '/root/mv_ca/dataset_full/clips_v1.5'
ann_raw_train = '/root/mv_ca/dataset_full/list_cvt/split_1/split1_train.txt'
ann_raw_val = '/root/mv_ca/dataset_full/list_cvt/split_1/split1_test.txt'
ann_save_train = '/root/mv_ca/dataset_full/train_mm.txt'
ann_save_val = '/root/mv_ca/dataset_full/validate_mm.txt'
calc_states = False
def dataset_prepare(raw_ann,save_ann,data_root):
    anns = []
    data_dict = {}
    with open(raw_ann,'r') as f:
        raw_lines = [x.strip().split('\t') for x in f.readlines()]
        frame_cnt_tot = 0
        min_frame_cnt = 100000
        max_frame_cnt = 0
        means = np.zeros(3)
        stds = np.zeros(3)
        for raw_ann in raw_lines:
            video_path = os.path.join(data_root,raw_ann[2])
            print('processing:',video_path)
            video_class = raw_ann[1]
            if video_class in data_dict.keys():
                data_dict[video_class] += 1
            else:
                data_dict[video_class] = 1
            anns.append([raw_ann[2],raw_ann[1]])
            if calc_states:
                video_cap = cv2.VideoCapture(video_path)
                frame_cnt = 0
                while True:
                    ret,frame = video_cap.read()
                    if not ret:break
                    frame_cnt += 1
                    for i in range(3):
                        means[i] += frame[:,:,i].mean()
                        stds[i] += frame[:,:,i].std()
                frame_cnt_tot += frame_cnt
                min_frame_cnt = min(frame_cnt,min_frame_cnt)
                max_frame_cnt = max(frame_cnt,max_frame_cnt)
        means /= frame_cnt_tot
        stds /= frame_cnt_tot
        print('frame cnt: avr:',frame_cnt_tot / len(raw_lines),'min:',min_frame_cnt,'max:',max_frame_cnt)
        print('classes:',data_dict)
        print('means:',means)
        print('stds:',stds)
    
    with open(save_ann,'w') as f:
        for ann in anns:
            f.write(str(ann[0]) + ' ' + str(int(ann[1]))+'\n')

dataset_prepare(ann_raw_val,ann_save_val,data_root_val)
dataset_prepare(ann_raw_train,ann_save_train,data_root_train)