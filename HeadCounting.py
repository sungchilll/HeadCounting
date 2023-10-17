import os
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound

# 디렉토리 생성
os.makedirs("inform", exist_ok=True)

# 음성 저장 및 재생 코드
text = '위험합니다! 다른 곳으로 이동하여 주십시오. '
text1 = '위험합니다! 해당 구역은 밀집도가 높으니 즉시 다른 곳으로 이동하여 주십시오.'
tts = gTTS(text=text, lang="ko")
tts.save("/home/user/detect/inform/danger.mp3")
tts1 = gTTS(text=text, lang="ko")
tts1.save("/home/user/detect/inform/danger2.mp3")

# 현재 스크립트의 절대 경로를 얻기 위해 os 모듈 사용
current_dir = os.path.dirname(os.path.abspath(__file__))
sound_file = os.path.join(current_dir, "/home/user/detect/inform/danger.mp3")
sound_file2 = os.path.join(current_dir, "/home/user/detect/inform/danger2.mp3")

# YOLO 모델 초기화
model = YOLO('yolov8n.pt')

# 카메라 스트리밍 및 객체 탐지
results = model(source=0, stream=True, classes=[0, 1], show=True)  # generator of Results objects

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs
   
    cnt = len(boxes.xyxy)
    if cnt > 0:
        playsound(sound_file)
        if cnt > 1:
        	playsound(sound_file2)
