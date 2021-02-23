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
  duration = 2000 * Fs
  audioTraining = np.zeros((duration,1), dtype = float)
  maskTraining = np.zeros((duration, 1), dtype = float)
  maxSilenceSegment = 2
  numSamples = 0
  file_ind = 0
  while numSamples < duration:
    if file_ind == len(files):
      break
    data, _ = librosa.load(os.path.join(path, files[file_ind]))
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


files = get_files(path)
audioTraining, maskTraining = dataCreate('/content/drive/MyDrive/Data/google_speech/train', files)
