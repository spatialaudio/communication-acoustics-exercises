"""Some tools used in the communication acoustics exercises."""
from __future__ import division  # Only needed for Python 2.x
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
try:
    from urllib.request import Request, urlopen  # Python 3.x
except ImportError:
    from urllib2 import Request, urlopen  # Python 2.x


def normalize(x, maximum=1, axis=None, out=None):
    """Normalize a signal to the given maximum (absolute) value.

    Parameters
    ----------
    x : array_like
        Input signal.
    maximum : float or sequence of floats, optional
        Desired (absolute) maximum value.  By default, the signal is
        normalized to +-1.0.  If a sequence is given, it must have the
        same length as the dimension given by `axis`.  Each sub-array
        along the given axis is normalized with one of the values.
    axis : int, optional
        Normalize along a given axis.
        By default, the flattened array is normalized.
    out : numpy.ndarray or similar, optional
        If given, the result is stored in `out` and `out` is returned.
        If `out` points to the same memory as `x`, the normalization
        happens in-place.

    Returns
    -------
    numpy.ndarray
        The normalized signal.

    """
    if axis is None and not np.isscalar(maximum):
        raise TypeError("If axis is not specified, maximum must be a scalar")

    maximum = np.max(np.abs(x), axis=axis) / maximum
    if axis is not None:
        maximum = np.expand_dims(maximum, axis=axis)
    return np.true_divide(x, maximum, out)


def fade(x, in_length, out_length=None, type='l', copy=True):
    """Apply fade in/out to a signal.

    If `x` is two-dimenstional, this works along the columns (= first
    axis).

    This is based on the *fade* effect of SoX, see:
    http://sox.sourceforge.net/sox.html

    The C implementation can be found here:
    http://sourceforge.net/p/sox/code/ci/master/tree/src/fade.c

    Parameters
    ----------
    x : array_like
        Input signal.
    in_length : int
        Length of fade-in in samples (contrary to SoX, where this is
        specified in seconds).
    out_length : int, optional
        Length of fade-out in samples.  If not specified, `fade_in` is
        used also for the fade-out.
    type : {'t', 'q', 'h', 'l', 'p'}, optional
        Select the shape of the fade curve: 'q' for quarter of a sine
        wave, 'h' for half a sine wave, 't' for linear ("triangular")
        slope, 'l' for logarithmic, and 'p' for inverted parabola.
        The default is logarithmic.
    copy : bool, optional
        If `False`, the fade is applied in-place and a reference to
        `x` is returned.

    """
    x = np.array(x, copy=copy)

    if out_length is None:
        out_length = in_length

    def make_fade(length, type):
        fade = np.arange(length) / length
        if type == 't':  # triangle
            pass
        elif type == 'q':  # quarter of sinewave
            fade = np.sin(fade * np.pi / 2)
        elif type == 'h':  # half of sinewave... eh cosine wave
            fade = (1 - np.cos(fade * np.pi)) / 2
        elif type == 'l':  # logarithmic
            fade = np.power(0.1, (1 - fade) * 5)  # 5 means 100 db attenuation
        elif type == 'p':  # inverted parabola
            fade = (1 - (1 - fade)**2)
        else:
            raise ValueError("Unknown fade type {0!r}".format(type))
        return fade

    # Using .T w/o [:] causes error: https://github.com/numpy/numpy/issues/2667
    x[:in_length].T[:] *= make_fade(in_length, type)
    x[len(x) - out_length:].T[:] *= make_fade(out_length, type)[::-1]
    return x


def db(x, power=False):
    """Convert a signal to decibel.

    Parameters
    ----------
    x : array_like
        Input signal.  Values of 0 lead to negative infinity.
    power : bool, optional
        If `power=False` (the default), `x` is squared before
        conversion.

    """
    with np.errstate(divide='ignore'):
        return 10 if power else 20 * np.log10(np.abs(x))


def blackbox(x, samplerate, axis=0):
    """Some unknown (except that it's LTI) digital system.

    Parameters
    ----------
    x : array_like
        Input signal.
    samplerate : float
        Sampling rate in Hertz.
    axis : int, optional
        The axis of the input data array along which to apply the
        system.  By default, this is the first axis.

    Returns
    -------
    numpy.ndarray
        The output signal.

    """
    # You are not supposed to look!
    b, a = signal.cheby1(8, 0.1, 3400 * 2 / samplerate)
    x = signal.lfilter(b, a, x, axis)
    b, a = signal.cheby1(4, 0.1, 300 * 2 / samplerate, 'high')
    return signal.lfilter(b, a, x, axis)


def blackbox_nonlinear(x, samplerate, axis=0):
    """Some unknown (except that it's non-linear) digital system.

    See Also
    --------
    blackbox

    """
    # You are not supposed to look!
    thr = 1/7
    out = blackbox(x, samplerate, axis)
    x = np.max(np.abs(out)) * thr
    return np.clip(out, -x, x, out=out)


def compressor(x, threshold, ratio, attack=0.03, release=0.003, makeup_gain=0):
    """ Compressor

    This is a python implementation of the Matlab file 'compexp.m' in
    Udo Zoelzer, Digitial Audio Signal Processing (Ch.4.2.2).
    The expander is omitted.

    Parameters
    ----------
    x : array-like
        Input signal.
    threshold : float
        Level in dB above which the compressor is active
    ratio : float
        Compression ratio (> 1)
    attack_time : float
        Attack time (> 0)
    release_time : float
        Release time (> 0)
    makeup_gain : float
        Make-up gain in dB to adjust the overall level

    """

    makeup_gain = 10**(makeup_gain / 20)  # convert to linear scale
    slope_factor = 1 - 1 / ratio
    tav = 0.01  # averaging time constant for level detection
    delay = 150
    xrms = 0
    g = 1
    buffer = np.zeros(delay)
    y = np.zeros(x.shape)

    for n in range(len(x)):
        xrms = (1 - tav) * xrms + tav * x[n]**2
        X = 10 * np.log10(xrms)
        G = np.min([0, slope_factor * (threshold - X)])
        f = 10**(G / 20)
        if f > g:
            coeff = attack
        else:
            coeff = release

        g = (1-coeff) * g + coeff * f
        y[n] = g * buffer[-1]
        buffer = np.concatenate(([x[n]], buffer[:-1:]))
    return makeup_gain * y


def edc(ir):
    L = len(ir)
    window = np.ones_like(ir)
    L = signal.fftconvolve(ir**2, window)[L-1:]
    return L / L[-1]


def rt20(ir, t0, fs=44100, plot=False):
    """Reverberation time RT20.

    Parameters
    ----------
    ir : array_type
        Room impulse response.
    t0 : float
        Reference time in milliseconds.
    fs : int, optional
        Sampling frequency.
    plot : bool, optional
        Plot the energy decay curve.

    Returns
    -------
    RT20 : float
        Reverberation time
    """

    L = edc(ir)
    n0 = int(np.round(t0 / 1000 * fs))  # Convert [ms] to [smaples]
    E0 = L[n0]  # Energy at the reference time t0
    n1 = int(np.argwhere(L < E0 * 10**-2)[0])  # 20 dB decay point
    T = 3 * (n1 - n0) / fs
    if plot:
        time = np.arange(len(L)) / fs * 1000
        t1 = n1 / fs * 1000
        E1 = L[n1]

        plt.figure(figsize=(10, 4))
        plt.plot(time, 10 * np.log10(L))
        plt.plot(time, -20 * (time - t0) / (t1 - t0) + 10 * np.log10(E0), 'r--')
        plt.plot(t0, 10 * np.log10(E0), 'o')
        plt.plot(t1, 10 * np.log10(E1), 'o')
        plt.xlabel('Time / ms')
        plt.ylabel('EDC / dB')
        plt.grid()
        plt.ylim(ymin=10 * np.log10(L[-1]))
        plt.title('RT = {:.2f} s'.format(T))
    return T


class HttpFile(object):
    """based on http://stackoverflow.com/a/7852229/500098"""

    def __init__(self, url):
        self._url = url
        self._offset = 0
        self._content_length = None

    def __len__(self):
        if self._content_length is None:
            response = urlopen(self._url)
            self._content_length = int(response.headers["Content-length"])
        return self._content_length

    def read(self, size=-1):
        request = Request(self._url)
        if size < 0:
            end = len(self) - 1
        else:
            end = self._offset + size - 1
        request.add_header('Range', "bytes={0}-{1}".format(self._offset, end))
        data = urlopen(request).read()
        self._offset += len(data)
        return data

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self._offset = offset
        elif whence == os.SEEK_CUR:
            self._offset += offset
        elif whence == os.SEEK_END:
            self._offset = len(self) + offset
        else:
            raise ValueError("Invalid whence")

    def tell(self):
        return self._offset
