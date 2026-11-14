import os
import sys
import json
import yaml
import shutil

#first you have to move ttpla_dataset into current dir
def test_existing():
    if os.path.exists("ttpla_dataset/splitting_dataset_txt"):
        ttpla_dataset = "ttpla_dataset/splitting_dataset_txt"
        if os.path.exists("data_original_size"):
            data_original_size = "data_original_size"
            print("Both dataset and splitting configs are ready")
        else:
            print("Splitting config is found, no dataset")
    elif os.path.exists("data_original_size"):
        print("Dataset found, no splitting config")
    else:
        print("No dataset and no splitting config")
    return ttpla_dataset, data_original_size

def get_configs_labels(labels, dataset):
    for file in os.listdir(labels):
        feature_class = file[:-3]
        shape_class = set()
        #print(feature_class)
        #print(f"{labels}/{file}")
        with open(f'{labels}/{file}', 'r', encoding='utf-8') as f:
            for line in f:
                _json = line.rstrip('\n')
                if not _json:
                    continue
                picture = _json.split(".")[0]+".jpg"
                src_path_label = os.path.join(f"{dataset}/", _json)
                with open (src_path_label, 'r') as _f:
                    raw_json = json.load(_f)
                    try:
                        for item in raw_json['shapes']:
                            #print(item['label'])
                            shape_class.add(item['label'])
                    except:
                        continue #if json is malformed
                    #print(raw_json['shapes']['label'])
                    #print(raw_json_label)
                src_path_picture = os.path.join(f"{dataset}/", picture)
                dst_path_label = os.path.join(f"dataset/labels/{feature_class}/", _json)
                dst_path_picture = os.path.join(f"dataset/images/{feature_class}/", picture)
                shutil.copy(src_path_label, dst_path_label)
                shutil.copy(src_path_picture, dst_path_picture)
    return shape_class
    
configs, dataset = test_existing()
shapes = get_configs_labels(configs, dataset)