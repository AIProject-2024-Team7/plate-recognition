class KoreaLicensePlateDataset(Dataset):
    def __init__(self, base_dir = 'data/', subset='train', transform=None): #subset -> train / valid / test
        
        self.image_dir = os.path.join(base_dir, subset, 'images')
        self.label_dir = os.path.join(base_dir, subset, 'labels')
        self.transform = transform

        # 이미지 파일 리스트 로드
        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    boxes.append([class_id] + bbox)

        if self.transform:
            image = self.transform(image)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return image, boxes
