import numpy as np
import math
import os
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

from scipy.io.wavfile import read as wavread
def ToolReadAudio(cAudioFilePath):    
    [samplerate, x] = wavread(cAudioFilePath)    
    if x.dtype == 'float32':        
        audio = x    
    else:        
        # change range to [-1,1)        
        if x.dtype == 'uint8':            
            nbits = 8        
        elif x.dtype == 'int16':            
            nbits = 16        
        elif x.dtype == 'int32':            
            nbits = 32        
        audio = x / float(2**(nbits - 1))    
        # special case of unsigned format    
    if x.dtype == 'uint8':        
        audio = audio - 1.    
    return (samplerate, audio)
def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * 
np.arange(iWindowLength)))
def compute_spectrogram(xb,fs):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    freq=np.fft.fftfreq(xb[0].size,1/fs)
    freqs = freq[:int(xb[0].size/2)+1]
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(np.fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]
        # freq=np.fft.fftfreq(xb[0].size,1/fs)
        # freqs[n]=freq[:int(xb[0].size/2)+1]
        # compute magnitude spectum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) 
    return X,freqs

def get_spectral_peaks(X):
    spectralPeaks = np.zeros([X.shape[1], 20])
    for i, block in enumerate(X.T):
        peaks = scipy.signal.find_peaks(block)[0]
        mag_peaks = block[peaks]
        max_peaks = np.argsort(mag_peaks)[-20:]
        spectralPeaks[i] = peaks[max_peaks]
    return spectralPeaks

def estimate_tuning_freq(x, blockSize, hopSize, fs):
    xb, tInSec = block_audio(x, blockSize, hopSize, fs)
    X, fInHz = compute_spectrogram(xb, fs)
    freq = get_spectral_peaks(X)
    numBlocks = X.shape[1]
    tf_audio = np.zeros(numBlocks*len(freq))

    def convert_freq2midi(fInHz, fA4InHz = 440):
        def convert_freq2midi_scalar(f, fA4InHz):
            if f <= 0:
                return 0
            else:
                return (69 + 12 * np.log2(f/fA4InHz))
        fInHz = np.asarray(fInHz)
        if fInHz.ndim == 0:
            return convert_freq2midi_scalar(fInHz,fA4InHz)
        midi = np.zeros(fInHz.shape)
        for k,f in enumerate(fInHz):
            midi[k] =  convert_freq2midi_scalar(f,fA4InHz)
        return (midi)

    for i, freq_block in enumerate(freq):
        # Find nearest temparament
        freq_MIDI = convert_freq2midi(freq_block)
        freq_MIDI_ref = np.round(freq_MIDI)
        # compute deltaC for a block
        deltaC_block = 100*(freq_MIDI - freq_MIDI_ref)
        # Store all delta c's.
        tf_audio[20*i:20*(i+1)] = deltaC_block
    hist, bins = np.histogram(tf_audio, bins = 10)
    # print (hist)
    center = (bins[:-1] + bins[1:]) / 2
    tf = center[np.argmax(hist)]
    tfInHz = 440*2**(tf/1200)
    return tfInHz

def convert_midi2freq(midi,fA4Hz = 440):
    def convert_midi2freq_scalar(p, fA4InHz):
        if p <= 0:
            return 0
        else:
            return fA4Hz*(2**((p-69)/12))#(69 + 12 * np.log2(f/fA4InHz))
    p = np.asarray(midi)
    if p.ndim == 0:
        return convert_midi2freq_scalar(p,fA4Hz)
    f=np.zeros(midi.shape)
    for k,p in enumerate(midi):
        f[k] = convert_midi2freq_scalar(p,fA4Hz)
    return f

def extract_pitch_chroma(X, fs, tfInHz):
    # C3 = 130.81Hz
    # B5 = 493.81Hz
    
    # Relationship between C3 and A4
    f = tfInHz*(2**(-21/12))

    pitch_classes = 12
    numOctaves = 3

    # initialise filterbank
    fb = np.zeros([pitch_classes, X.shape[0]])

    # Create filters for individual pitch classes
    for i in np.arange(pitch_classes):
        fb_tmp = np.zeros([X.shape[0]])
        for j in np.arange(numOctaves):
            # calculate the bounds for each and replace them with "1" in the actual filter
            f_tmp = f*2**j
            current_bin = f_tmp*2*(X.shape[0]-1)/fs
            bounds = np.array([2**(-1/(2*pitch_classes)), 2**(1/(2*pitch_classes))])*current_bin
            low = int(np.ceil(bounds[0]))
            high = int(np.ceil(bounds[1]))
            range = np.arange(low, high)
            nf = len(range)
            if len(range) == 0:
                range = low
                nf = 1
            # Normalize to the length of 1
            fb_tmp[range] = 1/nf
            j += 1
        fb[i] = fb_tmp
        f = f*2**(1/pitch_classes)
        i += 1

    pc = np.dot(fb, X**2)
    normalize = pc.sum(axis = 0)
    # Avoid division by 0
    normalize[normalize==0] = 1
    pc = pc/normalize

    return pc

def detect_key(x, blockSize, hopSize, fs, bTune = True):
    
    # Krumhansl
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                     [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    
    t_pc = t_pc / t_pc.sum(axis=1, keepdims=True)
    
    # key names
    KeyLable = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 12, 13, 14])
    
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    num_block, blockSize = xb.shape
    
    X, fInHz = compute_spectrogram(xb, fs)
    if bTune:
        tfInHz = estimate_tuning_freq(x, blockSize, hopSize, fs)
        pitchChroma = extract_pitch_chroma(X, fs, tfInHz)
    else:
        pitchChroma = extract_pitch_chroma(X, fs, 440)
    pitchChroma = pitchChroma.mean(axis=1)
    pitchChroma = np.concatenate((pitchChroma, pitchChroma), axis=0).reshape(2, 12)
    
    distance = np.zeros(t_pc.shape)
    
    for i in range(12):
        distance[:,i] = np.sum(np.abs(pitchChroma - np.roll(t_pc, i, axis=1)), axis=1)
        
    
    # minimum distance
    keyEstimate = KeyLable[distance.argmin()]
    
    return keyEstimate

def eval_tfe(pathToAudio, pathToGT):
    blockSize = 4096
    hopSize = 2048
    r = []
    for name in os.listdir(pathToAudio):
        if name.endswith(".wav"):
            name=name[:-4]
            sr,x = ToolReadAudio(pathToAudio+name+'.wav')
            lut = np.loadtxt(pathToGT+name+'.txt')
            tfHz = estimate_tuning_freq(x,blockSize,hopSize,sr)
            # print(name)
            # print(tfHz)
            # print(lut)
            # print('Difference: ',lut - tfHz)
            # print("-------------------------------------------")
            r.append(np.abs((lut - tfHz)))
    mae = np.array(r).mean()
    # print(mae)
    return mae

def eval_key_detection(pathToAudio, pathToGT):
    blockSize = 4096
    hopSize = 2048
    files = os.listdir(pathToAudio)
    for i, file in enumerate(files):
        if str(file)[-4:] != '.wav':
            files = np.delete(files, i)
            i += 0
        i += 1
    numFiles = len(files)

    estimated_keys = np.zeros(len(files))
    predicted_keys = np.zeros(len(files))
    for i, name in enumerate(files):
        fs, x = ToolReadAudio(pathToAudio+name)
        lut = np.loadtxt(pathToGT+name[:-4] + '.txt')
        predicted_keys[i] = lut
        estimated_keys[i] = detect_key(x, blockSize, hopSize, fs, bTune = True)

    diff = estimated_keys-predicted_keys
    incorrect = len(np.nonzero(diff)[0])
    # total = len(files)

    avg_accuracy = 1-incorrect/numFiles

    return avg_accuracy

def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):
    avg_accuracy = eval_key_detection(pathToAudioKey,pathToGTKey)
    avg_deviationInCent = eval_tfe(pathToAudioTf, pathToGTTf)

    return (avg_accuracy,avg_deviationInCent)

if __name__ == '__main__':
    complete_path_to_data_folder = '/Users/vedant/Desktop/Programming/ACA-assignments/aca-assignment-4/resources/key_tf'
    pathToAudioKey = complete_path_to_data_folder+'/key_eval/audio/'
    pathToGTKey = complete_path_to_data_folder+'/key_eval/GT/'

    pathToAudioTf = complete_path_to_data_folder+'/tuning_eval/audio/'
    pathToGTTf= complete_path_to_data_folder+'/tuning_eval/GT/'
    print(evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf))