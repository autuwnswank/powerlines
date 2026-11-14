from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="annotations",
    save_dir="converted",    
    cls91to80=False 
)