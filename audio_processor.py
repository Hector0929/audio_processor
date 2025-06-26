import os
import numpy as np
import math
from pydub import AudioSegment

class AudioSplitter:
    """
    一個用於根據 SNR 閾值自動切割音頻文件的類。
    在裁切前，音頻的響度將被標準化。
    """
    def __init__(self, snr_threshold_db: int = 30, 
                 frame_ms: int = 50, 
                 noise_estimation_ms: int = 500,
                 target_dbfs: float = -20.0, # 新增參數：目標響度 dBFS
                 output_dir: str = "output_segments_normalized"):
        """
        初始化 AudioSplitter 實例。

        Args:
            snr_threshold_db (int): 用於裁切的 SNR 閾值 (dB)。
            frame_ms (int): 用於計算 SNR 的幀長度（毫秒）。
            noise_estimation_ms (int): 用於估計噪聲基線的音頻開頭毫秒數。
            target_dbfs (float): 音頻正規化後的目標響度 (dBFS)。
            output_dir (str): 裁切後音頻片段的輸出目錄。
        """
        self.snr_threshold_db = snr_threshold_db
        self.frame_ms = frame_ms
        self.noise_estimation_ms = noise_estimation_ms
        self.target_dbfs = target_dbfs # 儲存目標響度
        self.output_dir = output_dir

        # 確保輸出目錄存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"創建輸出目錄: {self.output_dir}")

    def _calculate_rms(self, samples: np.ndarray) -> float:
        """
        計算音頻樣本的均方根 (RMS)。
        """
        if not samples.size:
            return 0.0
        return np.sqrt(np.mean(np.array(samples, dtype=np.float64)**2))

    def _calculate_snr(self, signal_plus_noise_rms: float, noise_rms: float) -> float:
        """
        根據信號加噪聲的 RMS 和純噪聲的 RMS 計算 SNR (dB)。
        """
        if noise_rms == 0:
            return float('inf')  # 避免除以零，表示無限大 SNR
        
        if signal_plus_noise_rms <= 0: # 信號加噪聲的 RMS 為零或負值 (理論上不應負)，SNR 無限小
            return -float('inf')

        ratio = signal_plus_noise_rms / noise_rms
        return 20 * np.log10(ratio)

    def _normalize_audio_to_target_loudness(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        將 AudioSegment 的響度標準化到一個目標 dBFS 值。
        這通常比將原始樣本的 RMS 調整為 1.0 更實用，因為它能讓音頻聽起來音量更正常。
        """
        current_loudness_dbfs = audio_segment.dBFS 
        
        if math.isinf(current_loudness_dbfs) and current_loudness_dbfs < 0: # 處理靜音音頻的 dBFS 為 -inf 的情況
            print("警告：音頻為靜音或接近靜音，無法有效標準化 dBFS。返回原始音頻。")
            return audio_segment
        
        # 使用 self.target_dbfs 作為目標響度
        gain_to_apply_db = self.target_dbfs - current_loudness_dbfs
        
        normalized_audio = audio_segment.apply_gain(gain_to_apply_db)
        print(f"音頻已標準化。原始 dBFS: {current_loudness_dbfs:.2f}, 調整增益: {gain_to_apply_db:.2f} dB, 最終 dBFS (目標: {self.target_dbfs:.2f}): {normalized_audio.dBFS:.2f}")
        return normalized_audio

    def split_audio(self, audio_path: str):
        """
        加載音頻文件，執行響度標準化，然後根據 SNR 閾值進行分割。

        Args:
            audio_path (str): 輸入音頻文件的路徑。
        """
        if not os.path.exists(audio_path):
            print(f"錯誤：音頻文件不存在於此路徑: {audio_path}")
            return

        try:
            print(f"\n--- 正在加載音頻文件: {audio_path} ---")
            audio = AudioSegment.from_file(audio_path)
            print("音頻加載成功。")
        except Exception as e:
            print(f"錯誤：無法加載音頻文件 {audio_path} - {e}")
            return

        print("\n--- 步驟 1: 標準化音頻響度 ---")
        # 調用新的響度標準化方法
        normalized_audio = self._normalize_audio_to_target_loudness(audio)
        
        # 獲取標準化後音頻的原始樣本數據
        data = np.array(normalized_audio.get_array_of_samples())
        frame_rate = normalized_audio.frame_rate
        
        print("\n--- 步驟 2: 估計噪聲 RMS ---")
        # 假設音頻開頭的一部分是純噪聲 (已經是標準化後的數據)
        noise_samples_count = int(frame_rate * (self.noise_estimation_ms / 1000.0))
        
        if noise_samples_count == 0:
            print("警告：噪聲估計時間太短，無法估計噪聲。請增加 noise_estimation_ms。")
            return
        
        # 確保不會超出音頻長度
        noise_samples_count = min(noise_samples_count, len(data))
        noise_data = data[:noise_samples_count]
        noise_rms = self._calculate_rms(noise_data)

        print(f"響度標準化後估計的噪聲 RMS (基線): {noise_rms:.6f}")

        if noise_rms == 0:
            print("警告：估計的噪聲 RMS 為零。這可能意味著音頻開頭沒有噪聲或文件有問題。將噪聲 RMS 設置為一個非常小的非零值。")
            noise_rms = 1e-9 # 設置一個非常小的非零值來避免後續計算問題

        frame_samples = int(frame_rate * (self.frame_ms / 1000.0))
        if frame_samples == 0:
            print("警告：幀長度太短，無法處理。請增加 frame_ms。")
            return

        current_segment_start_ms = None
        segments = []

        print(f"\n--- 步驟 3: 根據 SNR 閾值 ({self.snr_threshold_db} dB) 進行音頻分割 ---")

        for i in range(0, len(data), frame_samples):
            current_ms = (i / frame_rate) * 1000
            frame_data = data[i:i + frame_samples]

            if not frame_data.size: # 檢查幀是否為空，這可能發生在音頻末尾
                continue

            frame_rms = self._calculate_rms(frame_data)
            
            # 避免對數運算出現無效值
            if frame_rms <= 0:
                current_snr_db = -float('inf') 
            else:
                current_snr_db = self._calculate_snr(frame_rms, noise_rms)

            # 判斷是否滿足 SNR 閾值
            if current_snr_db >= self.snr_threshold_db:
                if current_segment_start_ms is None:
                    # 僅當不是在音檔最開頭的噪聲區間才開始新片段
                    # 檢查當前幀是否超過噪聲估計區間
                    if current_ms >= self.noise_estimation_ms:
                        current_segment_start_ms = current_ms
            else:
                if current_segment_start_ms is not None:
                    # 結束點設置為當前幀的起始點
                    segment_end_ms = current_ms 
                    segments.append((current_segment_start_ms, segment_end_ms))
                    current_segment_start_ms = None
        
        # 處理音頻末尾仍在裁切中的情況
        if current_segment_start_ms is not None:
            segments.append((current_segment_start_ms, len(normalized_audio)))

        print(f"\n--- 步驟 4: 保存裁切後的音頻片段 ---")
        if not segments:
            print("未檢測到符合 SNR 條件的音頻段。沒有文件被保存。")
            return

        print(f"檢測到 {len(segments)} 個符合條件的音頻段。")

        for idx, (start_ms, end_ms) in enumerate(segments):
            # 稍微擴展片段以包含更多上下文，避免語音被切掉一部分
            extension_ms = self.frame_ms # 擴展一個幀長度
            start_ms = max(0, start_ms - extension_ms) 
            end_ms = min(len(normalized_audio), end_ms + extension_ms) 
            
            chunk = normalized_audio[start_ms:end_ms]
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_filename = os.path.join(self.output_dir, f"{base_name}_{idx+1:02d}.wav")
            chunk.export(output_filename, format="wav")
            print(f"保存片段 {idx+1}: {start_ms:.0f} ms - {end_ms:.0f} ms 到 {output_filename}")

# --- 使用範例 ---
if __name__ == "__main__":
    # 這裡我們生成一個測試音頻，包含噪音、信號和結尾噪音，方便您測試
    from pydub.generators import Sine
    
    # 創建一個安靜的開始（低音量噪音），dBFS值會影響初始的響度
    noise_segment_test = AudioSegment.silent(duration=2000, frame_rate=44100).set_frame_rate(44100) - 30 # -30 dBFS 的靜音
    
    # 創建一個信號部分（低音量正弦波），dBFS值會影響初始的響度
    signal_segment_test = Sine(440).to_audio_segment(duration=3000, volume=-40).set_frame_rate(44100) # -40 dBFS 的正弦波
    
    # 將信號與一些背景噪音混合
    background_noise = AudioSegment.silent(duration=3000, frame_rate=44100).set_frame_rate(44100) - 60 # 更低的背景噪音
    mixed_signal_test = signal_segment_test.overlay(background_noise)
    
    # 創建一個安靜的結束
    end_noise_test = AudioSegment.silent(duration=2000, frame_rate=44100).set_frame_rate(44100) - 30
    
    test_audio_path_generated = "test_audio_for_class_example_loudness_norm.wav"
    (noise_segment_test + mixed_signal_test + end_noise_test).export(test_audio_path_generated, format="wav")
    print(f"已創建測試音頻文件: {test_audio_path_generated}")

    # 實例化 AudioSplitter
    # 您可以根據需要調整這些參數
    splitter = AudioSplitter(
        snr_threshold_db=30,      # SNR 閾值：高於 30dB 才裁切 (您可以嘗試降低這個值)
        frame_ms=10,              # 每 50 毫秒計算一次 SNR
        noise_estimation_ms=500,  # 使用音頻前 500 毫秒估計噪聲
        target_dbfs=-40.0,        # 標準化音頻到 -40 dBFS 的響度
        output_dir="segments_loudness_processed" # 輸出文件夾名稱
    )

    # 讓使用者輸入音檔位置
    while True:
        user_audio_path = input("\n請輸入您要處理的音頻文件路徑 (例如: my_audio.wav)，或輸入 'q' 退出: ")
        if user_audio_path.lower() == 'q':
            break
        
        splitter.split_audio(user_audio_path)
        print("\n--- 音頻處理完成 ---")

    print("程式結束。")