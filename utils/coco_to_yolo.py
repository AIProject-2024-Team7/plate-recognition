import json
import os

def convert_coco_to_yolo(coco_json_path, output_dir, image_dir):
    """
    COCO 포맷을 YOLO 포맷으로 변환.
    :param coco_json_path: COCO 포맷 JSON 파일 경로
    :param output_dir: YOLO 포맷 라벨 저장 경로
    :param image_dir: 이미지 파일이 저장된 경로
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 카테고리 ID 매핑
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # 각 이미지별 라벨 변환
    for img in coco_data['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_width = img['width']
        img_height = img['height']
        
        # YOLO 라벨 파일 생성
        label_file = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.txt")
        with open(label_file, 'w') as f:
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    category_id = categories[ann['category_id']]
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # YOLO 형식으로 변환
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
