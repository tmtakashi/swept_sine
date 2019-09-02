import numpy as np


def swept_sine(order=18, repeat=None):
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
    if repeat:
        s = np.r_[np.tile(s, repeat), np.zeros(N)]

    return s / np.max(np.abs(s)), inv_s / np.max(np.abs(inv_s))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import soundfile as sf

    fs = 48000
    repeat = 3
    order = 18
    tsp, itsp = swept_sine(order=order, repeat=repeat)
    sf.write('tsp_{}_{}.wav'.format(order, repeat), tsp, fs)
    sf.write('itsp_{}.wav'.format(order), itsp, fs)
