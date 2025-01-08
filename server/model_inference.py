import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt
from scipy.fft import fft

##################################################
# Lowpass-Filter-Funktionen (wie im Training)
##################################################

def butter_lowpass(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(window_df, cutoff=2, fs=15, order=2):
    """
    Wendet einen Butterworth-Tiefpassfilter (filtfilt) auf
    die Spalten ax, ay, az, gx, gy, gz an.
    """
    b, a = butter_lowpass(cutoff=cutoff, fs=fs, order=order)
    filtered_df = window_df.copy()
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        filtered_df[col] = filtfilt(b, a, window_df[col])
    return filtered_df

class DrinkDetectionModel:
    def __init__(self, model_path='drink_detection_model__kp.pkl'):
        self.model = joblib.load(model_path)
        self.USE_FFT = False  # Oder True, falls du FFT im Training genutzt hast

        # Parameter analog zum Training:
        self.cutoff = 2   # Grenzfrequenz
        self.fs = 15      # Abtastrate
        self.order = 2    # Filterordnung

    def compute_features(self, window_df):
        """
        Berechnet statistische Features für das gefilterte Datenfenster
        (30 Samples). Achtung: Wir nehmen die Spalten
        ax_filtered, ay_filtered, az_filtered, gx_filtered, gy_filtered, gz_filtered.
        """
        feature_dict = {}
        # Das entspricht deinem Training:
        accel_axes = ['ax_filtered', 'ay_filtered', 'az_filtered']
        gyro_axes  = ['gx_filtered', 'gy_filtered', 'gz_filtered']

        # Statistische Merkmale für Beschleunigung
        for axis in accel_axes:
            feature_dict[f'{axis}_mean'] = window_df[axis].mean()
            feature_dict[f'{axis}_std']  = window_df[axis].std()
            feature_dict[f'{axis}_max']  = window_df[axis].max()
            feature_dict[f'{axis}_min']  = window_df[axis].min()

        # Statistische Merkmale für Gyro
        for axis in gyro_axes:
            feature_dict[f'{axis}_mean'] = window_df[axis].mean()
            feature_dict[f'{axis}_std']  = window_df[axis].std()
            feature_dict[f'{axis}_max']  = window_df[axis].max()
            feature_dict[f'{axis}_min']  = window_df[axis].min()

        # Magnituden
        accel_mag = np.sqrt(window_df['ax_filtered']**2 + window_df['ay_filtered']**2 + window_df['az_filtered']**2)
        gyro_mag  = np.sqrt(window_df['gx_filtered']**2 + window_df['gy_filtered']**2 + window_df['gz_filtered']**2)
        feature_dict['accel_mag_mean'] = accel_mag.mean()
        feature_dict['accel_mag_std']  = accel_mag.std()
        feature_dict['gyro_mag_mean']  = gyro_mag.mean()
        feature_dict['gyro_mag_std']   = gyro_mag.std()

        # Optional: FFT Features
        if self.USE_FFT:
            # Beispiel: nur für ax_filtered
            fft_vals = np.abs(fft(window_df['ax_filtered']))
            half_n = len(fft_vals) // 2
            feature_dict['ax_fft_mean'] = fft_vals[:half_n].mean()
            feature_dict['ax_fft_std']  = fft_vals[:half_n].std()

        # DataFrame mit 1 Zeile erstellen
        X_df = pd.DataFrame([feature_dict])
        return X_df

    def predict(self, window_samples):
        """
        Nimmt 30 Roh-Samples (List[Dict]) entgegen,
        1) wandelt sie in ein DataFrame um
        2) wendet Lowpass-Filter an
        3) benennt Spalten in *_filtered um
        4) berechnet Features
        5) macht die Vorhersage
        """
        # 1) In DataFrame umwandeln
        df_window = pd.DataFrame(window_samples)

        # 2) Lowpass-Filter anwenden
        filtered_df = apply_lowpass_filter(
            df_window,
            cutoff=self.cutoff,
            fs=self.fs,
            order=self.order
        )

        # 3) Spalten passend umbenennen, damit sie heißen wie beim Training:
        #    ax => ax_filtered, ay => ay_filtered, usw.
        filtered_df.rename(
            columns={
                "ax": "ax_filtered",
                "ay": "ay_filtered",
                "az": "az_filtered",
                "gx": "gx_filtered",
                "gy": "gy_filtered",
                "gz": "gz_filtered"
            },
            inplace=True
        )

        # 4) Feature-Berechnung
        X = self.compute_features(filtered_df)

        # 5) Vorhersage (RF-Klassifizierung)
        pred = self.model.predict(X)[0]  # 0 oder 1
        return int(pred)
