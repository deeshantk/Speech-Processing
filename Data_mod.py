import librosa
import os
import random

'''
Shuffle the data 
'''

def get_files(path):
  files = []
  for file in os.listdir(path):
    for audio in os.listdir(os.path.join(path, file)):
      files.append(file + '/' + audio)
  random.shuffle(files)
  return files


'''
Creating a 1000-secound training signal by combining speech files and random period of silence between them.
'''

def dataCreate(path, files):
  duration = 1000 * Fs
  audioTraining = np.zeros((duration,1), dtype = float)
  maskTraining = np.zeros((duration, 1), dtype = float)
  maxSilenceSegment = 2
  numSamples = 0
  file_ind = 0
  while numSamples < duration:
    if file_ind == len(files):
      break
    data, _ = librosa.load(os.path.join(path, files[file_ind]), 16000)
    data = data / max(abs(data))
    idx = detectSpeech(data, Fs)
    try:
      data = data[idx[0]:idx[1]+1]
      audioTraining[numSamples:numSamples+len(data)] = data.reshape(len(data), -1)
      maskTraining[numSamples:numSamples + len(data)] = True
      numSilenceSamples = np.random.randint(1, maxSilenceSegment*Fs)
      numSamples += len(data) + numSilenceSamples;
    except Exception as e:
      pass
    file_ind += 1
  return audioTraining, maskTraining

'''
Adding noise to training data such that SNR is -5 dB.
'''
def addNoise(audioTraining, noise):
  noise = noise.reshape(len(noise), -1)
  noise = noise[:len(audioTraining)]
#  audioTraining = audioTraining[:len(noise)]
  SNR = -5
  noise = (10**(-SNR/20)) * noise * np.linalg.norm(audioTraining) / np.linalg.norm(noise)
  #noise = noise / max(abs(noise))
  audioTrainingNoisy = audioTraining + noise
  audioTrainingNoisy = audioTrainingNoisy / max(abs(audioTrainingNoisy))
  return audioTrainingNoisy

'''
Creating noise of length at least 1000 secounds to add in our training data.  
'''
def noiseProcess(noise, length):
  noise, sr = librosa.load(noise, 16000)
  total_len = (1/sr) * len(noise)
  while total_len < length:
    noise = np.append(noise, noise)
    total_len = (1/sr) * len(noise)
 # noise = noise[:length]
  return noise


files = get_files(path) #Add path 
audioTraining, maskTraining = dataCreate(training_path, files)    #Add path
noise = noiseProcess(noise_path, 1000)      #Add path
audioTrainingNoisy = addNoise(audioTraining, noise)
