"""Shared preprocessing used by training, visualization and inference."""
import base64,io,math
import numpy as np
import torch
import soundfile as sf
from PIL import Image,ImageOps
from scipy import signal
from scipy.fft import dct
import torch.nn.functional as F
SR=16000
SECONDS=5
AUDIO_VERSION='audio-16k-bandpass100-2000-mel64-mfcc20-v1'
XRAY_VERSION='xray-gray224-autocontrast-v1'

def load_audio(source):
    try:
        f=sf.SoundFile(source)
    except (sf.LibsndfileError,RuntimeError):
        raise ValueError('Unsupported or corrupt audio; use PCM WAV or FLAC.') from None
    with f:
        if f.frames/f.samplerate>120: raise ValueError('Audio must be no longer than 120 seconds.')
        if f.frames/f.samplerate<1: raise ValueError('Record at least one second of audio.')
        sr=f.samplerate
        y=f.read(dtype='float32',always_2d=True).mean(axis=1)
    if not np.isfinite(y).all(): raise ValueError('Audio contains invalid sample values.')
    if sr!=SR:
        g=math.gcd(sr,SR); y=signal.resample_poly(y,SR//g,sr//g)
    if float(np.max(np.abs(y)))<1e-6: raise ValueError('Audio is silent; record a clear respiratory sound.')
    return y.astype(np.float32)

def clean_audio(y):
    sos=signal.butter(4,[100,2000],btype='bandpass',fs=SR,output='sos')
    y=signal.sosfiltfilt(sos,y).astype(np.float32)
    _,_,z=signal.stft(y,fs=SR,nperseg=512,noverlap=352)
    magnitude=np.abs(z); floor=np.percentile(magnitude,10,axis=1,keepdims=True)
    z=z*np.maximum(magnitude-floor,0)/(magnitude+1e-10)
    _,y=signal.istft(z,fs=SR,nperseg=512,noverlap=352)
    peak=np.max(np.abs(y)); return (y/max(float(peak),1e-8)).astype(np.float32)

def mel_spectrogram(y):
    _,_,z=signal.stft(y,fs=SR,nperseg=512,noverlap=352)
    power=np.abs(z)**2
    to_mel=lambda hz:2595*np.log10(1+hz/700)
    to_hz=lambda mel:700*(10**(mel/2595)-1)
    points=to_hz(np.linspace(to_mel(100),to_mel(2000),66))
    freqs=np.linspace(0,SR/2,257);bank=np.zeros((64,257),np.float32)
    for i in range(64): bank[i]=np.maximum(0,np.minimum((freqs-points[i])/(points[i+1]-points[i]),(points[i+2]-freqs)/(points[i+2]-points[i+1])))
    mel=10*np.log10(np.maximum(bank@power,1e-10)); return mel.astype(np.float32)

def audio_tensor(segment):
    segment=np.pad(segment,(0,max(0,SR*SECONDS-len(segment))))[:SR*SECONDS]
    mel=mel_spectrogram(segment)
    mfcc=dct(mel,type=2,axis=0,norm='ortho')[:20]
    def scale(a):return (a-a.mean())/(a.std()+1e-6)
    a=torch.from_numpy(scale(mel))[None,None]
    b=torch.from_numpy(scale(mfcc))[None,None]
    return torch.cat([F.interpolate(a,size=(64,128),mode='bilinear',align_corners=False),F.interpolate(b,size=(64,128),mode='bilinear',align_corners=False)],dim=1)[0]

def png64(image):
    out=io.BytesIO();image.save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()

def audio_preview(y):
    mel=mel_spectrogram(y)
    indices=np.linspace(0,len(y)-1,min(len(y),1200)).astype(int)
    values=(np.clip((mel+80)/80,0,1)*255).astype(np.uint8)
    rgb=np.stack([np.clip(values.astype(float)*1.8-150,0,255),np.clip(values.astype(float)*1.4-40,0,255),np.clip(210-values.astype(float)*.5,0,255)],axis=-1).astype(np.uint8)
    img=Image.fromarray(np.flipud(rgb)).resize((640,200))
    return {'waveform':np.round(y[indices],5).tolist(),'duration':round(len(y)/SR,2),'spectrogram':png64(img),'sampleRate':SR}

def load_xray(source):
    image=Image.open(source)
    if image.format not in ['PNG','JPEG']:raise ValueError('Use a PNG or JPEG image.')
    if image.width<64 or image.height<64 or image.width*image.height>25000000:raise ValueError('Image dimensions must be at least 64×64 and at most 25 megapixels.')
    return ImageOps.exif_transpose(image).convert('L')

def xray_tensor(image):
    # Autocontrast supplies the specified contrast enhancement. Median filter reduces isolated noise.
    from PIL import ImageFilter
    image=ImageOps.autocontrast(image).filter(ImageFilter.MedianFilter(3)).resize((224,224))
    a=np.asarray(image,dtype=np.float32)/255
    return torch.from_numpy((a-.5)/.5)[None]
