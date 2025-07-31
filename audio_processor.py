import sys
import subprocess
import importlib.util

def check_and_install_libraries():
    """
    檢查必要的 Python 套件是否已安裝，若未安裝則自動使用 pip 安裝。
    """
    # 定義必要套件列表
    required_libraries = {
        "parselmouth": "praat-parselmouth", # 安裝時的名稱可能與導入時不同
        "numpy": "numpy",
        "scipy": "scipy"
    }
    
    print("正在檢查必要的套件...")
    for lib_import_name, lib_install_name in required_libraries.items():
        spec = importlib.util.find_spec(lib_import_name)
        if spec is None:
            print(f"'{lib_import_name}' 套件未安裝。正在嘗試自動安裝...")
            try:
                # 使用 sys.executable 確保為當前 Python 環境安裝
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib_install_name])
                print(f"'{lib_install_name}' 已成功安裝。")
            except subprocess.CalledProcessError as e:
                print(f"錯誤：安裝 '{lib_install_name}' 失敗。請手動執行 'pip install {lib_install_name}'。")
                print(f"錯誤細節: {e}")
                sys.exit(1) # 安裝失敗則退出程式
    print("所有必要套件均已安裝。\n")

# --- 在導入套件前，先執行檢查與安裝 ---
check_and_install_libraries()

# 現在可以安全地導入套件
import parselmouth
import numpy as np
import scipy.fftpack
import scipy.stats

class Audio_processor:
    """
    一個用於處理和分析音訊檔案的類別，專為咳嗽等生理訊號設計。
    根據 'ACCOUGH 聲學特徵分析' 文件中定義的參數進行分析。
    """
    
    def __init__(self, file_path):
        """
        初始化 Audio_processor。
        :param file_path: 要分析的音訊檔案路徑 (例如 'C1_reflex_1.wav')。
        """
        self.file_path = file_path
        self.sound = None
        self.sound_normalized = None
        self.results = {}
        
        self._load_sound()

    def _load_sound(self):
        """
        從指定路徑載入音訊檔案。
        """
        try:
            self.sound = parselmouth.Sound(self.file_path)
        except parselmouth.PraatError:
            print(f"錯誤：無法載入音檔 '{self.file_path}'。請確認檔案路徑是否正確。")
            raise

    def _energy_normalize(self):
        """
        對音訊進行能量正規化，使其平均能量為 1。
        這是一個內部輔助函式。
        """
        # 獲取樣本，此時 shape 為 (聲道數, 取樣點數)，例如 (1, N)
        samples = self.sound.values
        mean_energy = np.mean(samples**2)
        
        if mean_energy == 0:
            # 如果音訊是靜音，則不進行正規化
            self.sound_normalized = self.sound
            return

        normalization_factor = np.sqrt(1 / mean_energy)
        normalized_samples = samples * normalization_factor
        
        # 使用正規化後的樣本建立新的 Sound 物件
        # 確保 normalized_samples 的 shape 仍然是 (聲道數, 取樣點數)
        self.sound_normalized = parselmouth.Sound(values=normalized_samples, sampling_frequency=self.sound.sampling_frequency)

    def get_length(self):
        """計算音訊的總長度（秒）。"""
        return self.sound_normalized.get_total_duration()

    def get_crest_factor(self):
        """計算波峰因數及其相對位置。"""
        # 建議：使用 .flatten() 確保為一維陣列
        samples = self.sound_normalized.values.flatten()
        if not samples.any():
            return 0.0, 0.0
            
        rms = np.sqrt(np.mean(samples**2))
        if rms == 0:
            return 0.0, 0.0
            
        max_abs_val = np.max(np.abs(samples))
        crest_factor = max_abs_val / rms
        
        peak_position_index = np.argmax(np.abs(samples))
        relative_position = peak_position_index / len(samples)
        
        return crest_factor, relative_position

    def analyze_amplitude_contour(self):
        """分析振幅輪廓，回傳斜率 (DCT-II) 與曲率 (DCT-III)。"""
        intensity = self.sound_normalized.to_intensity()
        intensity_values = np.nan_to_num(intensity.values[0])
        
        if len(intensity_values) < 4:
            return 0.0, 0.0
            
        dct_coeffs = scipy.fftpack.dct(intensity_values, type=2, norm='ortho')
        
        slope = dct_coeffs[1] if len(dct_coeffs) > 1 else 0.0
        curvature = dct_coeffs[2] if len(dct_coeffs) > 2 else 0.0
        
        return slope, curvature

    def get_kurtosis(self):
        """計算整個訊號的峰度 (Pearson's definition)。"""
        # 建議：使用 .flatten() 確保為一維陣列
        samples = self.sound_normalized.values.flatten()
        return scipy.stats.kurtosis(samples, fisher=False)

    def get_sample_entropy(self, m=2, r_ratio=0.2):
        """
        計算樣本熵的簡化實現。
        注意：這是一個佔位符實現。
        """
        return np.random.rand() * 2

    def get_relative_energy_in_bands(self):
        """計算各頻帶的相對能量。"""
        spectrum = self.sound_normalized.to_spectrum()
        # 直接計算能量總和
        total_energy = np.sum(spectrum.values**2)
        if total_energy == 0:
            return {}

        bands = {
            "0-400Hz": (0, 400),
            "400-800Hz": (400, 800),
            "800-1600Hz": (800, 1600),
            "1600-3200Hz": (1600, 3200),
            "3200Hz-Nyquist": (3200, self.sound.sampling_frequency / 2)
        }
        
        relative_energies = {}
        for name, (fmin, fmax) in bands.items():
            # 找出頻率落在該頻帶的索引
            freq_array = spectrum.xs()
            freq_indices = np.where((freq_array >= fmin) & (freq_array < fmax))[0]
            # 計算該頻帶的能量
            band_energy = np.sum(spectrum.values[0][freq_indices] ** 2)
            relative_energies[name] = (band_energy / total_energy) * 100
            
        return relative_energies

    def run_analysis(self):
        """
        執行完整的聲學分析流程。
        """
        if not self.sound:
            print("音訊未載入，無法分析。")
            return

        print(f"開始分析，載入的音訊長度為: {self.sound.get_total_duration()} 秒")    
        self._energy_normalize()
        
        length = self.get_length()
        crest_factor, rel_pos_crest_factor = self.get_crest_factor()
        amp_slope, amp_curvature = self.analyze_amplitude_contour()
        kurtosis = self.get_kurtosis()
        sample_entropy = self.get_sample_entropy()
        relative_energies = self.get_relative_energy_in_bands()

        # 建議：儲存原始浮點數值
        self.results = {
            "Length": length,
            "Amplitude contour slope": amp_slope,
            "Amplitude contour curvature": amp_curvature,
            "Sample entropy contour": sample_entropy,
            "Kurtosis contour": kurtosis,
            "Crest factor": crest_factor,
            "Relative position of crest factor": rel_pos_crest_factor,
            **{f"Energy {k} (%)": v for k, v in relative_energies.items()}
        }
        return self.results

    def display_results(self):
        """
        以 Markdown 表格格式顯示分析結果。
        """
        if not self.results:
            print("沒有可顯示的結果。請先執行 .run_analysis()")
            return
        
        file_name = self.file_path.split('/')[-1].split('\\')[-1]
        
        print(f"\n### 分析結果報表: `{file_name}`")
        print(f"| 項目名稱 | {file_name}_01 |")
        print("|---|---|")
        # 建議：在顯示時才格式化
        for key, value in self.results.items():
            if "Energy" in key:
                print(f"| {key} | {value:.2f} |")
            else:
                print(f"| {key} | {value:.4f} |")
        print("\n> **注意**: Sample entropy contour 的數值為隨機佔位符。")


if __name__ == "__main__":
    import os
    print(f"腳本當前的工作目錄是: {os.getcwd()}")
    
    file_path_input = input("請輸入要分析的音訊檔案路徑 (例如: C1_reflex_1.wav): ")
    
    absolute_path = os.path.abspath(file_path_input)
    print(f"正在嘗試載入絕對路徑: {absolute_path}")
    print(f"該路徑是否存在: {'是' if os.path.exists(absolute_path) else '否'}")

    # 如果檔案不存在，提前退出
    if not os.path.exists(absolute_path):
        print("錯誤：檔案不存在於指定路徑，請檢查您的工作目錄或輸入的相對/絕對路徑。")
    else:
        try:
            audio_analyzer = Audio_processor(file_path_input)
            audio_analyzer.run_analysis()
            audio_analyzer.display_results()
            
        except Exception as e:
            print(f"處理過程中發生錯誤: {e}")
