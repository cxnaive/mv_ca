import torch
import numpy as np
from mmaction.apis import init_recognizer, inference_recognizer

class_labels = 'datasets/mapping_table.txt'
with open(class_labels,'r') as f:
    class_list = np.array([x.strip().split('\t') for x in f.readlines()])

config_file = 'mmaction2/configs/recognition/csn/ircsn_dark_dataset.py'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)
chk_point_file = '/root/mv_ca/mmaction2/work_dirs/ircsn_dark_dataset/epoch_58.pth'

model = init_recognizer(config_file,chk_point_file, device=device)
print('init done.')
# inference the demo video
results = inference_recognizer(model, 'Drink_3_2.mp4')
for res in results:
    print(res)
    print(class_list[res[0]])