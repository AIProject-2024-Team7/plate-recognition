from ultralytics import YOLO
from sklearn.model_selection import GridSearchCV
import numpy as np

# YOLO Wrapper
class YOLOModelWrapper:
    def __init__(self, model_path='yolov8n.pt', data='data.yaml'):
        self.model_path = model_path
        self.data = data
        self.results = None
        self.params = {}

    def fit(self, X=None, y=None, **params):

        model = YOLO('yolov8n.pt') 
        self.params = params
        model.train(
            data='./detection.yaml',        
            epochs=200,               
            batch=params.get('batch', 16),
            lr0 = params.get('lr', 0.01),                
            imgsz=640,               
            optimizer='Adam',        
            project='license_plate_noisy_v5_gridsearch', 
            name='yolov8_finetuned_detection_noisy_v5',  
            device=0,
            patience=20                 
        )
        self.results = model.val()

    def score(self, X=None, y=None):
        metrics = self.results
        print(metrics)
        return metrics['metrics/mAP50']
        
    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self
    

param_grid = {
    'batch': [8, 16],
    'lr': [0.005, 0.01],
}


def main():
    yolo_wrapper = YOLOModelWrapper()
    grid_search = GridSearchCV(
        estimator=yolo_wrapper,
        param_grid=param_grid,
        scoring=yolo_wrapper.score,
        cv=2,               
    )

    grid_search.fit([1,2,3,4,5]) # dummy
    print("Best Parameters:", grid_search.best_params_)

if __name__ == '__main__':
    main()