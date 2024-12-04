import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

# 데이터 경로 설정
train_dir = 'C:/Users/User/Desktop/pet/train'
valid_dir = 'C:/Users/User/Desktop/pet/valid'
test_dir = 'C:/Users/User/Desktop/pet/test'

# 하이퍼파라미터 설정
image_size = (224, 224)  # 입력 이미지 크기
batch_size = 32  # 배치 크기
num_classes = 3  # 클래스 수 (예: 3개의 클래스)

# 데이터 증강 및 로드
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # 픽셀 값을 [0,1]로 정규화
    rotation_range=20,  # 이미지를 20도 회전
    width_shift_range=0.2,  # 이미지를 수평으로 20% 이동
    height_shift_range=0.2,  # 이미지를 수직으로 20% 이동
    shear_range=0.2,  # 이미지를 시계 방향으로 전단
    zoom_range=0.2,  # 이미지를 20% 확대/축소
    horizontal_flip=True  # 이미지를 좌우 반전
)

# 검증 및 테스트 데이터는 증강 없이 스케일링만 적용
valid_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,  # 모든 이미지를 (224x224)로 크기 변경
    batch_size=batch_size,
    class_mode='categorical'  # 다중 클래스 원-핫 인코딩
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Confusion Matrix 계산을 위해 순서 고정
)

# VGG16 모델 구성
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 사전 학습된 가중치 고정

# 사용자 정의 레이어 추가
model = Sequential([
    base_model,  # VGG16 모델의 Feature Extractor
    Flatten(),  # 출력 텐서를 1D로 변환
    Dense(256, activation='relu'),  # Fully Connected Layer
    Dropout(0.5),  # 과적합 방지를 위한 Dropout
    Dense(num_classes, activation='softmax')  # 클래스별 확률 계산
])

# 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Adam 옵티마이저 사용
    loss='categorical_crossentropy',  # 다중 클래스 분류 손실 함수
    metrics=['accuracy']  # 정확도를 평가 지표로 설정
)

# 콜백 설정: EarlyStopping과 ReduceLROnPlateau
early_stopping = EarlyStopping(
    monitor='val_loss',  # 검증 손실을 기준으로 조기 종료
    patience=5,  # 5번 연속으로 개선되지 않으면 중단
    restore_best_weights=True  # 최상의 가중치를 복원
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # 검증 손실을 기준으로 학습률 조정
    factor=0.5,  # 학습률을 절반으로 감소
    patience=3,  # 3번 연속으로 개선되지 않으면 학습률 감소
    min_lr=1e-7  # 학습률의 하한값 설정
)

# 모델 학습
history = model.fit(
    train_generator,  # 훈련 데이터
    validation_data=valid_generator,  # 검증 데이터
    epochs=50,  # 최대 50번 반복
    steps_per_epoch=train_generator.samples // batch_size,  # 한 에포크에서 훈련 배치 수
    validation_steps=valid_generator.samples // batch_size,  # 한 에포크에서 검증 배치 수
    callbacks=[early_stopping, reduce_lr]  # 콜백 리스트
)

# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")  # 테스트 정확도 출력

# Confusion Matrix 및 Classification Report
def evaluate_model(test_generator, model):
    Y_pred = model.predict(test_generator)  # 예측값
    y_pred_classes = np.argmax(Y_pred, axis=1)  # 예측된 클래스
    y_true = test_generator.classes  # 실제 클래스 레이블
    class_labels = list(test_generator.class_indices.keys())  # 클래스 이름 추출

    # Confusion Matrix 계산 및 시각화
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.title("Confusion Matrix with Values")
    plt.show()

    # Classification Report 출력
    report = classification_report(y_true, y_pred_classes, target_names=class_labels, digits=4)
    print("Classification Report:")
    print(report)

# ROC Curve
def plot_roc_curve(test_generator, model):
    Y_pred = model.predict(test_generator)  # 예측값
    y_true = to_categorical(test_generator.classes, num_classes=num_classes)  # 실제 클래스 레이블 원-핫 인코딩

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(test_generator.class_indices.keys()):
        fpr, tpr, _ = roc_curve(y_true[:, i], Y_pred[:, i])  # ROC Curve 계산
        roc_auc = auc(fpr, tpr)  # AUC 계산
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')  # 무작위 추측선
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# 평가 실행
evaluate_model(test_generator, model)  # Confusion Matrix와 Classification Report
plot_roc_curve(test_generator, model)  # ROC Curve 시각화

# Validation Accuracy Plot
def plot_accuracy(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')  # 훈련 정확도
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # 검증 정확도
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_accuracy(history)  # 정확도 시각화