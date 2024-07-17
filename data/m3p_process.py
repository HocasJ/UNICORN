import numpy as np
from PIL import Image
import pdb
import os

import easydict
import yaml

args = yaml.load(open('config/config_sysu.yaml'), Loader=yaml.FullLoader)
args = easydict.EasyDict(args)

data_path = args.data_path#'/root/mmmp1_10/'
#data_path = '/root/sysu/'

###---------------需要修改--------------
exp_setting = args.exp_setting
# rgb_cameras = ['01','02','03','04','05','06']
# ir_cameras = ['14']
setting_name_split = exp_setting.split("_")
print("setting_name_split:",setting_name_split)

if setting_name_split[1] == 'cctv' and setting_name_split[2] == 'ir' and setting_name_split[3] == 'cctv' and setting_name_split[4] == 'rgb':
    rgb_cameras = ['07','08','09','10','11','12']
    ir_cameras = ['01','02','03','04','05','06']
elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'ir' and setting_name_split[3] == 'cctv' and setting_name_split[4] == 'rgb':
    rgb_cameras = ['14']
    ir_cameras = ['01','02','03','04','05','06']
elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'rgb' and setting_name_split[3] == 'cctv' and setting_name_split[4] == 'ir':
    rgb_cameras = ['13']
    ir_cameras = ['07','08','09','10','11','12']
elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'ir' and setting_name_split[3] == 'cctv' and setting_name_split[4] == 'ir':
    rgb_cameras = ['14']
    ir_cameras = ['07','08','09','10','11','12']
elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'ir' and setting_name_split[3] == 'uav' and setting_name_split[4] == 'rgb':
    rgb_cameras = ['14']
    ir_cameras = ['13']
elif setting_name_split[1] == 'uav' and setting_name_split[2] == 'rgb' and setting_name_split[3] == 'cctv' and setting_name_split[4] == 'rgb':
    rgb_cameras = ['13']
    ir_cameras = ['01','02','03','04','05','06']
else:
    print("!!!setting name error!!!!")

# load id info
file_path_train = os.path.join(data_path,exp_setting,'train_id.txt')#os.path.join(data_path,'exp_uav_ir_cctv_rgb/train_id.txt')
file_path_val   = os.path.join(data_path,exp_setting,'val_id.txt')#
###---------------需要修改--------------

with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]
    
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
    
# combine train and val split   
id_train.extend(id_val) 

files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)
            
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

# relabel
pid_container = set()
for img_path in files_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
pid2label = {pid:label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 384
def read_imgs(train_image):
    train_img = []
    train_label = []
    train_path = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array) 
        
        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)
         # path
        train_path.append(img_path)
    return np.array(train_img), np.array(train_label), np.array(train_path)
       
# rgb imges
train_img, train_label, train_path = read_imgs(files_rgb)
np.save(data_path + exp_setting + 'train_rgb_resized_img.npy', train_img)
np.save(data_path + exp_setting + 'train_rgb_resized_label.npy', train_label)
np.save(data_path + exp_setting + 'train_rgb_resized_path.npy', train_path)
# ir imges
train_img, train_label, train_path = read_imgs(files_ir)
np.save(data_path + exp_setting + 'train_ir_resized_img.npy', train_img)
np.save(data_path + exp_setting +'train_ir_resized_label.npy', train_label)
np.save(data_path + exp_setting +'train_ir_resized_path.npy', train_path)

