# OCR Recognition 코드
import torch
import torch.nn as nn

class CRNNRecognizer(nn.Module):
    #CRNN 기반 번호판 텍스트 인식 모델
    def __init__(self, input_channels=3, num_classes=36, hidden_size=256):
        super(CRNNRecognizer, self).__init__()
        
        # CNN 계층
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # RNN 계층
        self.rnn = nn.LSTM(256, hidden_size, bidirectional=True, batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Bidirectional RNN이므로 hidden_size*2

    def forward(self, x):
        
        #입력 번호판 이미지를 받아 텍스트를 예측.
        #:param x: 번호판 이미지 텐서
        #:return: 클래스 확률 분포
        
        x = self.cnn(x)  # CNN으로 이미지 특징 추출
        x = x.view(x.size(0), x.size(3), -1)  # (Batch, Width, Channels)
        x, _ = self.rnn(x)  # RNN으로 시퀀스 처리
        x = self.fc(x)  # Fully Connected Layer로 클래스 예측
        return x


