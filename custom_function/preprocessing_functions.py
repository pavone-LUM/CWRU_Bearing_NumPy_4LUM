import matplotlib.pyplot as plt
import numpy as np
import math

import numpy as np


def custom_fft(signal, fs=12000):
    N = len(signal)
    # FFT
    spectrum_full = np.fft.fft(signal)
    # Frequency axis
    freqs_full = np.fft.fftfreq(N, d=1 / fs)
    # Keep only positive half
    half = N // 2
    freqs = freqs_full[:half]
    # spectrum = spectrum_full[:half]
    spectrum = np.abs(spectrum_full[:half]) / N  # modulo normalizzato

    # return freqs, spectrum
    return  spectrum


def split_signal_into_segments(signal, w):
    n = len(signal)
    M = n // w                     # how many full segments
    trimmed = signal[:M * w]       # remove leftover tail
    segments = trimmed.reshape(M, w)
    return segments


def split_and_fft(signal, w, fs=12000):
    """
    Split in segments of length w and apply custom FFT to each.
    Returns a matrix (M × w/2) where M = n // w.
    """
    n = len(signal)
    M = n // w
    trimmed = signal[:M * w]
    segments = trimmed.reshape(M, w)

    # Apply FFT to each segment
    fft_list = []

    for seg in segments:
        spectrum = custom_fft(seg, fs)
        fft_list.append(spectrum)

    return np.vstack(fft_list)

import numpy as np

def average_over_window(arr, aw):
    rows, cols = arr.shape
    M = cols // aw                     # number of full blocks
    trimmed = arr[:, :M * aw]         # remove leftover columns
    reshaped = trimmed.reshape(rows, M, aw)
    out = reshaped.mean(axis=2)       # average inside each block

    return out

import pandas as pd
import numpy as np

def rand_train_test_split(df, test_size=0.2, random_state=42):
    """
    Randomly split a dataframe into train and test subsets.

    Parameters:
        df (DataFrame): Input dataframe
        test_size (float): Fraction of rows to assign to test set (0 to 1)
        random_state (int): Seed for reproducibility

    Returns:
        train_df, test_df
    """
    np.random.seed(random_state)

    # Shuffle the index
    shuffled_idx = np.random.permutation(df.index)

    # Compute split point
    test_count = int(len(df) * test_size)

    # Split
    test_idx = shuffled_idx[:test_count]
    train_idx = shuffled_idx[test_count:]

    # Return dataframes
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)
