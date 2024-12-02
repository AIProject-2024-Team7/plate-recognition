from ultralytics import YOLO
from matplotlib import rc
import matplotlib.font_manager as fm
import yaml

font_path = './fonts/NANUMGOTHIC.TTF'
font_prop = fm.FontProperties(fname=font_path)

rc('font', family=font_prop.get_name())

def main():
    model = YOLO('yolov8n.pt')  

    model.train(
        data='./recognition.yaml',  
        epochs=500,                      
        batch=16,                        
        imgsz=640,                       
        optimizer='Adam',                
        project='recognition',   
        name='yolov8_finetuned_recognition',         
        device=0,
        split=0.9,
        patience=300                         
    )

    metrics = model.val()
    print(metrics)  
 
if __name__ == '__main__':
    main()


