import numpy as np


def swept_sine(order=18):
    N = int(2 ** order)
    m = N / 4
    S = np.zeros(N, dtype=np.complex128)
    k = np.arange(N / 2 + 1)

    S[:N // 2 + 1] = np.exp(-1j * 4 * m * np.pi * k ** 2 / N ** 2)

    S[N // 2 + 1:] = np.conj(S[1: N // 2][::-1])

    iS = 1 / S

    s = np.fft.irfft(S, N)
    inv_s = np.fft.irfft(iS, N)

    shift = int(N / 2 - m)

    s = np.roll(s, shift)
    inv_s = np.r_[inv_s[shift:], inv_s[:shift]]
    return s / np.max(np.abs(s)), inv_s / np.max(np.abs(inv_s))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import soundfile as sf

    fs = 48000
    tsp, itsp = swept_sine(order=18)
    sf.write('tsp18.wav', tsp, fs)
    sf.write('itsp18.wav', itsp, fs)
