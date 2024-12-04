import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 경로 설정
train_dir = 'C:/Users/User/Desktop/pet/train'  # 훈련 데이터 경로
valid_dir = 'C:/Users/User/Desktop/pet/valid'  # 검증 데이터 경로
test_dir = 'C:/Users/User/Desktop/pet/test'    # 테스트 데이터 경로

# 이미지 크기 및 배치 크기 설정
img_size = (299, 299)  # Xception 모델은 299x299 이미지 크기를 요구
batch_size = 32  # 배치 크기

# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 픽셀 값을 [0, 1]로 정규화
    rotation_range=30,  # 이미지를 최대 30도 회전
    width_shift_range=0.2,  # 이미지를 수평으로 20% 이동
    height_shift_range=0.2,  # 이미지를 수직으로 20% 이동
    shear_range=0.2,  # 전단 변환
    zoom_range=0.3,  # 이미지를 최대 30% 확대/축소
    horizontal_flip=True,  # 이미지를 좌우 반전
    fill_mode='nearest'  # 빈 픽셀을 가장 가까운 값으로 채움
)

# 검증 및 테스트 데이터는 증강 없이 스케일링만 적용
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=1, class_mode='categorical', shuffle=False
)

# 모델 정의 (Xception 사용)
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = True  # 사전 학습된 가중치를 일부 수정 가능하도록 설정

# 특정 레이어까지만 학습 가능하도록 제한
for layer in base_model.layers[:100]:
    layer.trainable = False  # 초기 100개의 레이어는 동결

# 사용자 정의 모델 구성
model = Sequential([
    base_model,  # Xception 사전 학습 모델
    GlobalAveragePooling2D(),  # 전역 평균 풀링 레이어
    Dropout(0.5),  # 50%의 뉴런 Dropout으로 과적합 방지
    Dense(256, activation='relu'),  # Fully Connected Layer
    Dropout(0.5),  # 추가 Dropout
    Dense(train_generator.num_classes, activation='softmax')  # 최종 출력 레이어 (다중 클래스)
])

# 모델 컴파일
optimizer = Adam(learning_rate=0.0001)  # 학습률 설정
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_loss',  # 검증 손실을 기준으로 조기 종료
    patience=5,  # 5번 연속으로 개선되지 않으면 종료
    restore_best_weights=True  # 최적의 가중치를 복원
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # 검증 손실을 기준으로 학습률 감소
    factor=0.2,  # 학습률을 20%로 감소
    patience=3,  # 3번 연속 개선되지 않으면 감소
    min_lr=1e-6  # 학습률의 최소값 설정
)

# 모델 학습
history = model.fit(
    train_generator,
    epochs=30,  # 최대 30 에포크 학습
    validation_data=valid_generator,
    callbacks=[early_stopping, reduce_lr]  # 콜백 사용
)

# 학습 정확도 시각화
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')  # 훈련 정확도
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # 검증 정확도
plt.legend()
plt.title('Training and Validation Accuracy')  # 제목
plt.xlabel('Epochs')  # x축 레이블
plt.ylabel('Accuracy')  # y축 레이블
plt.show()

# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")  # 테스트 정확도 출력

# 예측 수행
test_steps = len(test_generator)
test_generator.reset()
y_pred_prob = model.predict(test_generator, steps=test_steps)  # 예측 확률
y_pred = np.argmax(y_pred_prob, axis=1)  # 예측된 클래스
y_true = test_generator.classes  # 실제 클래스 레이블

# Confusion Matrix 생성 및 시각화
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_mat, annot=True, fmt='d', cmap='Blues',
    xticklabels=test_generator.class_indices.keys(),
    yticklabels=test_generator.class_indices.keys()
)
plt.title('Confusion Matrix')  # 제목
plt.xlabel('Predicted')  # 예측 레이블
plt.ylabel('Actual')  # 실제 레이블
plt.show()

# Classification Report 생성
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# ROC Curve 계산 및 시각화
fpr = {}
tpr = {}
roc_auc = {}
n_classes = len(test_generator.class_indices)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_generator.classes == i, y_pred_prob[:, i])  # ROC Curve 계산
    roc_auc[i] = auc(fpr[i], tpr[i])  # AUC 계산

plt.figure(figsize=(10, 8))
for i, label in enumerate(test_generator.class_indices.keys()):
    plt.plot(fpr[i], tpr[i], label=f'Class {label} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')  # 랜덤 추측선
plt.xlabel('False Positive Rate')  # x축 레이블
plt.ylabel('True Positive Rate')  # y축 레이블
plt.title('ROC Curve')  # 제목
plt.legend(loc='lower right')  # 범례 위치
plt.show()