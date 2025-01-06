import numpy as np
import argparse
import librosa

# 定义按键频率表
DTMF_FREQUENCIES = {
    (697, 1209): '1', (697, 1336): '2', (697, 1477): '3',
    (770, 1209): '4', (770, 1336): '5', (770, 1477): '6',
    (852, 1209): '7', (852, 1336): '8', (852, 1477): '9',
    (941, 1209): '*', (941, 1336): '0', (941, 1477): '#'
}

def detect_key(frequencies):
    """根据频率组合检测按键"""
    precision = 10  # 精度
    for (f1, f2), key in DTMF_FREQUENCIES.items():
        if abs(f1 - frequencies[0]) < precision and abs(f2 - frequencies[1]) < precision:
            return key
    return -1

def frame_signal(signal, frame_length, hop_length):
    """对信号进行分帧"""
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_length)[::hop_length]
    return frames[:num_frames]

def key_tone_recognition(audio_data):
    y, sr = audio_data
    frame_length = int(sr / 64)
    hop_length = frame_length

    # 计算短时能量和动态阈值
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten()
    threshold = np.percentile(energy, 10)  # 动态设定静默阈值

    # 分帧处理
    frames = frame_signal(y, frame_length, hop_length)
    hann_window = np.hanning(frame_length)  # Hann 窗函数

    results = []
    for i, frame in enumerate(frames):
        if energy[i] < threshold:
            results.append(-1)
        else:
            # 对帧加窗后计算 FFT
            windowed_frame = frame * hann_window
            spectrum = np.abs(np.fft.rfft(windowed_frame, n=2048))
            frequencies = np.fft.rfftfreq(2048, d=1/sr)

            # 找到两个主频率
            peak_indices = spectrum.argsort()[-2:][::-1]  # 提取前两大频率
            detected_freqs = frequencies[peak_indices]
            detected_freqs.sort()
            key = detect_key(detected_freqs)
            results.append(key)

    return " ".join(map(str, results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type=str, help='test file name', required=True)
    args = parser.parse_args()
    input_audio_array = librosa.load(args.audio_file, sr=48000, dtype=np.float32)  # 加载音频文件
    output = key_tone_recognition(input_audio_array)
    print(output)
