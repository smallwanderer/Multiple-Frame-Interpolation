import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import options
import interpolator

# 가상의 입력 데이터 생성을 위한 파라미터 설정
batch_size = 1
height, width, channels = 64, 64, 3
time_steps = 4

# tf.keras.Input을 사용해 입력 텐서 정의
x0_input = tf.keras.Input(shape=(height, width, channels), name='x0')
x1_input = tf.keras.Input(shape=(height, width, channels), name='x1')
time_input = tf.keras.Input(shape=(time_steps,), name='time')

# 모델 설정 불러오기
config = options.Options()  # options.py에 정의된 Options 클래스 인스턴스화
config.use_aux_outputs = True  # 보조 출력 활성화

# 모델 생성
model = interpolator.create_model(x0_input, x1_input, time_input, config)

# 가상의 입력 데이터 생성
x0 = tf.random.uniform(shape=(batch_size, height, width, channels), minval=0, maxval=1, dtype=tf.float32)
x1 = tf.random.uniform(shape=(batch_size, height, width, channels), minval=0, maxval=1, dtype=tf.float32)
time_values = np.linspace(0, 1, time_steps + 2)[1:-1]
time = tf.constant(time_values.reshape(1, -1).repeat(batch_size, axis=0), dtype=tf.float32)

# 모델 예측 수행
outputs = model({'x0': x0, 'x1': x1, 'time': time})

# 결과 확인 및 출력 구조 검사
print("Model outputs:")
for key, value in outputs.items():
    if isinstance(value, list):  # 리스트(피라미드 구조)인 경우
        print(f"{key}: List of tensors with shapes:")
        for idx, tensor in enumerate(value):
            print(f"  Level {idx}: shape = {tensor.shape}")
    else:  # 단일 텐서인 경우
        print(f"{key}: shape = {value.shape}")

# 1. 각 시간 단계별로 생성된 중간 프레임의 평균 및 분산 계산
print("Mean and variance of each intermediate frame:")
for i in range(time_steps):
    img_array = outputs[f'image_t{i}'].numpy().squeeze()
    mean_val = np.mean(img_array)
    var_val = np.var(img_array)
    print(f"Time step {i}: Mean = {mean_val:.4f}, Variance = {var_val:.4f}")

# 2. 각 중간 프레임 간의 차이 계산
print("\nDifference between consecutive frames:")
for i in range(1, time_steps):
    img_array_prev = outputs[f'image_t{i - 1}'].numpy().squeeze()
    img_array_curr = outputs[f'image_t{i}'].numpy().squeeze()
    diff = np.abs(img_array_curr - img_array_prev)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    print(f"Difference between time step {i - 1} and {i}: Mean diff = {mean_diff:.4f}, Max diff = {max_diff:.4f}")

# 3. 특정 좌표 픽셀 값 비교 (예: 이미지 중앙 좌표)
center_coords = (height // 2, width // 2)
print("\nCenter pixel values across time steps:")
for i in range(time_steps):
    img_array = outputs[f'image_t{i}'].numpy().squeeze()
    center_pixel_val = img_array[center_coords[0], center_coords[1], :]
    print(f"Time step {i} - Center pixel RGB values: {center_pixel_val}")

# 보조 출력 값도 확인 (보조 출력이 활성화된 경우)
if config.use_aux_outputs:
    for i in range(time_steps):
        x0_warped = outputs[f'x0_warped_t{i}']
        x1_warped = outputs[f'x1_warped_t{i}']
        print(f"x0_warped_t{i}: shape = {x0_warped.shape}")
        print(f"x1_warped_t{i}: shape = {x1_warped.shape}")
