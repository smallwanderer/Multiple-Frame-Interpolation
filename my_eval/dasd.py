import numpy as np
import tensorflow as tf
from my_eval import util

# 모델 로드
model_path = "C:\\Users\\McGra\\frame-interpolation\\mnist_train\\my_mnist_train_03\\saved_model"  # 실제 모델 경로로 변경하세요.
model = tf.saved_model.load(model_path)

# 테스트 프레임 로드
frame0_path = '/photos/test_no_01/frames_interpolate\\01\\frame0.png'  # 실제 frame0 경로로 변경하세요.
frame19_path = '/photos/test_no_01/frames_interpolate\\01\\frame1.png'  # 실제 frame19 경로로 변경하세요.

frame0 = util.read_image(frame0_path)
frame19 = util.read_image(frame19_path)

# 다양한 dt 값 설정
time_values = np.linspace(0, 1, 10, endpoint=False)  # 0부터 1 사이의 10개 dt 값

# 중간 프레임 생성 및 저장
for i, dt in enumerate(time_values):
    # 입력 데이터 준비
    inputs = {
        'x0': np.expand_dims(frame0, axis=0),  # 배치 차원 추가
        'x1': np.expand_dims(frame19, axis=0),  # 배치 차원 추가
        'time': np.array([[dt]], dtype=np.float32)  # 중간 시간 값 (배치 포함, (1,1) 형태)
    }

    # 모델로 중간 프레임 생성
    output = model(inputs)
    interpolated_frame = output['image'][0].numpy()  # 첫 번째 배치 결과 추출

    # 결과 프레임 저장
    output_path = f'interpolated_frame_{i:02d}_dt_{dt:.2f}.png'
    util.write_image(output_path, interpolated_frame)
    print(f"Saved interpolated frame for dt={dt:.2f} at: {output_path}")