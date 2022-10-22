import os
import json
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision import models
from tqdm import tqdm

scene_path = "../../data/public"
image_path = "../../data/public_image"
meta_path = "../../data"
n_turns = 2

mode = "test"

if mode == "train":
    dial_path = "../../data/simmc2.1_dials_dstc11_train.json"    
    output_path = "./data/object_f1_train_turn2.txt"

else:
    dial_path = "../../data/simmc2.1_dials_dstc11_devtest.json"    
    output_path = "./data/object_f1_devtest_turn2_with_pred.txt"


if mode == "train":
    fashion_model = None
    furniture_model = None
else:
    fashion_model = models.resnext101_32x8d(pretrained=False)
    n_features = fashion_model.fc.in_features
    fashion_model.fc = nn.Linear(n_features, 251)

    fashion_model.load_state_dict(torch.load('./fashion_devtest_baseline.bin'))
    fashion_model.eval()

    furniture_model = models.resnext101_32x8d(pretrained=False)
    n_features = furniture_model.fc.in_features
    furniture_model.fc = nn.Linear(n_features, 30)

    furniture_model.load_state_dict(torch.load('./furniture_devtest_baseline.bin'))
    furniture_model.eval()




with open("./idx2fashion_id.json") as f:
    idx2fashion = json.load(f)

with open("./idx2furniture_id.json") as f:
    idx2furniture = json.load(f)

class2prefab = {}

with open(os.path.join(meta_path, "fashion_prefab_metadata_all.json")) as f:
    meta_data = json.load(f)
    
    for prefab_i in meta_data.keys():
        class2prefab[prefab_i.replace("/", "_")] = prefab_i

with open(os.path.join(meta_path, "furniture_prefab_metadata_all.json")) as f:
    meta_data = json.load(f)
    
    for prefab_i in meta_data.keys():
        class2prefab[prefab_i.replace("/", "_")] = prefab_i

with open(dial_path) as f:
    devtest_dial = json.load(f)
    
    
def scene2object(scenes, domain):
    if domain == "fashion":
        meta_file = os.path.join(meta_path, "fashion_prefab_metadata_all.json")
        model = fashion_model
        
        mean_nums = [0.274, 0.258, 0.259] # [0.485, 0.456, 0.406] -> [0.274, 0.258, 0.259]
        std_nums = [0.207, 0.200, 0.204] # [0.229, 0.224, 0.225] -> [0.207, 0.200, 0.204]
        
        id2class = idx2fashion
        
        vis_meta = ["assetType", "color", "pattern"]

    else:
        meta_file = os.path.join(meta_path, "furniture_prefab_metadata_all.json")
        model = furniture_model
        
        mean_nums = [0.504, 0.470, 0.438] # [0.485, 0.456, 0.406] -> [0.504, 0.470, 0.438]
        std_nums = [0.297, 0.287, 0.274] # [0.229, 0.224, 0.225] -> [0.297, 0.287, 0.274]

        id2class = idx2furniture

        vis_meta = ["color", "type"]

    test_transform = T.Compose([
        T.ToTensor(),
        T.Resize(size=[256,256]),
        T.CenterCrop(size=256),
        T.Normalize(mean_nums, std_nums)
    ])
        
    scene_ids = scenes.items()
    scene_objects_data = {}
    
    for scene_num, scene_id in scene_ids:
        scene_i_path = os.path.join(scene_path, scene_id + "_scene.json")
        
        image_i_id = scene_id.replace("m_cloth", "cloth")
        image_i_path = os.path.join(image_path, image_i_id + ".png")
        
        with open(scene_i_path) as f:
            scene_i_data = json.load(f)
        
        for scene_info in scene_i_data["scenes"]:
            # print(scene_data_i)
            # scene_info = scene_data_i[0]

            objects_i = scene_info['objects']
            # relationships_i = scene_info['relationships']
            
            for object_i in objects_i:
                try:
                    scene_objects_data[object_i['index']].append({'prefab_path': object_i['prefab_path'], 'bbox': object_i['bbox'],
                                                                  'position': object_i['position'], 'image_path': image_i_path, 'turn': scene_num})
                except:
                    scene_objects_data[object_i['index']] = [{'prefab_path': object_i['prefab_path'], 'bbox': object_i['bbox'],
                                                              'position': object_i['position'], 'image_path': image_i_path, 'turn': scene_num}]


    objects_meta = {}

    with open(meta_file) as f:
        meta_data = json.load(f)
    
    for object_index in scene_objects_data.keys():
        total_logit = None
        min_turn = 9999
        
        if mode != "train":
            model.cuda()
            with torch.no_grad():
                for object_index_info in scene_objects_data[object_index]:
                    scene_img = Image.open(object_index_info['image_path'])
                    bbox = object_index_info['bbox']
                    turn = int(object_index_info['turn'])
                    
                    if int(turn) < min_turn:
                        min_turn = turn
                    
                    if bbox[2] < 2:
                        bbox[1] = max(0, bbox[1] - 2)
                        bbox[2] = 4
                    
                    if bbox[3] < 2:
                        bbox[0] = max(0, bbox[0] - 2)
                        bbox[3] = 4
                    
                    # object에 해당하는 부분만 crop 및 tensor로 변환

                    object_img = scene_img.crop((bbox[0], bbox[1], bbox[0] + bbox[3], bbox[1] + bbox[2]))
                    image = np.array(object_img.convert('RGB'))
                    rgb_img = np.float32(image) / 255

                    input_tensor = test_transform(rgb_img).unsqueeze(0)
                    input_tensor = input_tensor.cuda()
                    
                    output_logit = model(input_tensor)
                    
                    if total_logit is None:
                        total_logit = output_logit
                    else:
                        total_logit += output_logit

                total_logit = total_logit[0].data
                # print(torch.argmax(total_logit).item(), bbox)
                
            pred_prefab = class2prefab[id2class[str(torch.argmax(total_logit).item())]]
                
            pred_meta = meta_data[pred_prefab]
            
            ans_prefab = scene_objects_data[object_index][0]['prefab_path']
            ans_meta = meta_data[ans_prefab]
            
            # logit으로 prefab 예측
            # 정답 prefab로부터 meta data 가져오기
                
            final_meta_data = {k:v for k,v in ans_meta.items()}
            final_meta_data.update({k:pred_meta[k] for k in vis_meta})
            
            objects_meta[object_index] = final_meta_data
            
            model.cpu()
        else:
            for object_index_info in scene_objects_data[object_index]:
                turn = int(object_index_info['turn'])
                
                if int(turn) < min_turn:
                    min_turn = turn
                
            ans_prefab = scene_objects_data[object_index][0]['prefab_path']
            ans_meta = meta_data[ans_prefab]
            objects_meta[object_index] = ans_meta
    
        objects_meta[object_index].update({"turn":min_turn})
        
    return objects_meta


def dial2data(data_i):
    dial_i = data_i['dialogue']
    domain = data_i['domain']
    scenes = data_i['scene_ids']
    
    meta_dict = scene2object(scenes, domain)

    transcript_list = []
    system_list = []
    label_list = []
    mentioned_object_list = []

    prev_metioned_object = set()
    for turn_i in dial_i:
        mentioned_object_list.append(list(prev_metioned_object))

        transcript_list.append(turn_i['transcript'])
        system_list.append(turn_i['system_transcript'])
        label_list.append(turn_i['transcript_annotated']['act_attributes']['objects'])
        prev_metioned_object.update(turn_i['transcript_annotated']['act_attributes']['objects'])
        prev_metioned_object.update(turn_i['system_transcript_annotated']['act_attributes']['objects'])
    
    return transcript_list, system_list, label_list, meta_dict, mentioned_object_list

total_data = []

for idx in tqdm(range(len(devtest_dial["dialogue_data"]))):
    t_list, s_list, l_list, m_dict, o_list = dial2data(devtest_dial["dialogue_data"][idx])
    
    total_data.append({'transcript': t_list, 'system_transcript': s_list, 'labels': l_list, 'meta': m_dict, 'mentioned_object' : o_list})
    


MENTIONED_OBJECT = "<MO>"
NOT_MENTIONED_OBJECT = "<NMO>"
USER_TURN = "<UT>"
SYSTEM_TURN = "<ST>"


json_data = total_data
    
data = []

for data_i in json_data: # 발화
    n = len(data_i['transcript'])
    
    for idx in range(n): # Turn
        
        dial_list = []
        
        for i in range(min(n_turns - 1, idx)):
            dial_list.append(USER_TURN + " " + data_i['transcript'][idx + 1 - n_turns - i])
            dial_list.append(SYSTEM_TURN + " " + data_i['system_transcript'][idx + 1 - n_turns - i])
        
        dial_list.append(USER_TURN + " " + data_i['transcript'][idx])
        
        dial_i = " ".join(dial_list)
        label_i = data_i['labels'][idx]
        mentioned_list = data_i['mentioned_object'][idx]
        
        tmp = 1
        
        if len(label_i) == 0:
            tmp = 0
        
        for object_id in data_i['meta'].keys():
            object_meta = data_i['meta'][object_id]
            
            if object_meta['turn'] < idx:
                label = 0
                
                if int(object_id) in label_i:
                    label = 1
                    
                if int(object_id) in mentioned_list:
                    is_mentioned = MENTIONED_OBJECT
                else:
                    is_mentioned = NOT_MENTIONED_OBJECT
                    
                meta_keys = sorted(list(object_meta.keys()))
                
                if "availableSizes" in meta_keys:
                    meta_keys.remove("availableSizes")
                meta_keys.remove("turn")

                meta_list = [str(meta_key) + " is " + str(object_meta[meta_key]) + "." for meta_key in meta_keys]
                meta_str = " ".join(meta_list)
                
                # dial + meta + label + object_id + empty_label
                # 이전 
                data_str = "\t".join([dial_i, meta_str, str(label), str(object_id), str(tmp), is_mentioned]) + "\n"
                
                if mode == "train":
                    if label == 1:
                        for i in range(10):
                            data.append(data_str)
                    else:
                        data.append(data_str)
                else:
                    data.append(data_str)
                
                
with open(output_path, "w") as f:
    for data_i in data:
        f.write(data_i)