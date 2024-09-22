## YOLO와 face_recognition 라이브러리를 활용한 얼굴 인식 및 얼굴 출입여부 표시 시스템
- YOLO 모델이 얼굴 검출을 담당하고, face_recognition이 얼굴 인코딩을 담당하는 방식
  
### 1. 주요기능
#### 1) 얼굴 등록
- load_encodings(): JSON 파일에서 기존 얼굴 인코딩을 로드합니다.
- save_encodings(): 얼굴 인코딩을 JSON 파일에 저장합니다.
- register_face(): 새로운 얼굴을 등록합니다.
- open_file_dialog(): 파일 선택 대화상자를 열어 이미지를 선택합니다.

#### 2) 얼굴 인식
- recognize_faces(): 웹캠을 사용하여 실시간으로 얼굴을 인식합니다.
- extract_embedding(): 얼굴 이미지에서 임베딩을 추출합니다.
- draw_text_centered_above_ellipse(): 화면에 텍스트를 표시합니다.

#### 3) GUI 구성
Tkinter를 사용하여 간단한 GUI를 생성합니다.
사용자 이름 입력 필드, 얼굴 등록 버튼, 얼굴 인식 버튼을 포함합니다.

### 2. face_recognition 라이브러리
#### 1) 얼굴 등록 과정
```bash
def register_face(name, image_path):
    image = face_recognition.load_image_file(image_path)  # 이미지 파일을 로드
    encodings = face_recognition.face_encodings(image)    # 로드된 이미지에서 얼굴을 검출하고 인코딩
```

#### 2) 얼굴 인식 과정
```bash
# 실시간 비디오 스트림에서 검출된 얼굴에 대해 인코딩을 수행
def extract_embedding(face_image):
    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_face)
```

#### 3) 얼굴 비교
```bash
# face_recogintion 라이브러리로 생성된 인코딩을 사용하여 코사인 유사도를 계산합니다.
sim = cosine_similarity([face_embedding], [embedding])[0][0]
```

### 3. YOLO 모델 학습
- 캐글의 [Face-Detection-Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)을 YOLO 모델에 추가 학습
```bash
# YOLOv8 모델 로드 
model = YOLO('yolov8n.pt')   # YOLOv8의 가장 작은 버전인 'nano' 모델

# 모델 학습
model.train(
    data='data.yaml',  # YOLOv8 데이터셋 설정 파일 경로
    epochs=50,         # 에포크 수
    imgsz=640,         # 이미지 크기
    batch=16,          # 배치 크기
    name='face_detection_yolov8',  # 모델 저장 이름
)
```
