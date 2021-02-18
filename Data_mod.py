'''

Creating a 1000-secound training signal by combining speech files and random period of silence between them.

'''


duration = 2000 * Fs
audioTraining = np.zeros((duration,1), dtype = float)
maskTraining = np.zeros((duration, 1), dtype = float)

maxSilenceSegment = 2

numSamples = 0
l = []
while numSamples < duration:
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
  break
