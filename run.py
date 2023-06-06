import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sys


# Training (Functional Method)
model = keras.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size =(3, 3), activation='relu',input_shape=(64, 64, 1)))
model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(152, activation = 'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(76))
model.add(keras.layers.Dropout(0.2))

# Connect heads to final output layer
model.add(keras.layers.Dense(19*6, activation = 'softmax'))
model.add(keras.layers.Reshape((6,19)))

model.compile(optimizer = keras.optimizers.Adam(lr=0.01), loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.load_weights("final_model.h5")

def audio_CQT(audio_path, start, dur):  # start and dur in seconds
    
    # Function for removing noise
    def cqt_lim(CQT):
        new_CQT = np.copy(CQT)
        new_CQT[new_CQT < -60] = -120
        return new_CQT
    
    # Perform the Constant-Q Transform
    data, sr = librosa.load(audio_path, sr = None, mono = True, offset = start, duration = dur)
    CQT = librosa.cqt(data, sr = 44100, hop_length = 1024, fmin = None, n_bins = 96, bins_per_octave = 12)
    CQT_mag = librosa.magphase(CQT)[0]**4
    CQTdB = librosa.core.amplitude_to_db(CQT_mag, ref = np.amax)
    new_CQT = cqt_lim(CQTdB)
    return new_CQT

def auto_tab(wav):
    predicted_tabs = []
    final_tab = ''
    y,sr=librosa.load(wav)
    dur = librosa.get_duration(y=y,sr=sr) 
    for j in np.arange(0,dur,0.2):
        C = audio_CQT(wav,j,0.2)
        fig, ax = plt.subplots()
        plt.axis('off')
        img = librosa.display.specshow(C, x_axis='time', y_axis='cqt_note', ax=ax, cmap='gray_r')
        plt.savefig("test.png")
        plt.close()
        img = tf.keras.preprocessing.image.load_img('test.png',color_mode='grayscale',target_size=(64, 64))
        arr = tf.keras.preprocessing.image.img_to_array(img).reshape(64,64)/255
        tab = model.predict(np.array([arr]))
        for i in tab[0]:
            ir=np.array(i)
            a = np.zeros_like(ir, dtype=float)  # Create an array of zeros with the same shape as arr
            max_index = np.argmax(ir)  # Get the index of the largest value in arr
            a[max_index] = 1
            a.astype(int).tolist()
            predicted_tabs.append(a)

    tabs_sub = [predicted_tabs[i:i+120] for i in range(0, len(predicted_tabs), 120)]

    for tabs in tabs_sub:
        for i in range(5,len(tabs),6):
            if len(np.nonzero(tabs[i]==1)[0])!=0:
                index = np.nonzero(tabs[i]==1)[0][0] - 1
            if index != -1: 
                if index < 10: final_tab += '---'+str(index)+'--'
                else: final_tab += '--'+str(index)+'--'
                continue
            final_tab += '------'
        final_tab+='\n'
        for i in range(4,len(tabs),6):
            if len(np.nonzero(tabs[i]==1)[0])!=0:
                index = np.nonzero(tabs[i]==1)[0][0] - 1
            if index != -1: 
                if index < 10: final_tab += '---'+str(index)+'--'
                else: final_tab += '--'+str(index)+'--'
                continue
            final_tab += '------'
        final_tab+='\n'
        for i in range(3,len(tabs),6):
            if len(np.nonzero(tabs[i]==1)[0])!=0:
                index = np.nonzero(tabs[i]==1)[0][0] - 1
            if index != -1: 
                if index < 10: final_tab += '---'+str(index)+'--'
                else: final_tab += '--'+str(index)+'--'
                continue
            final_tab += '------'
        final_tab+='\n'
        for i in range(2,len(tabs),6):
            if len(np.nonzero(tabs[i]==1)[0])!=0:
                index = np.nonzero(tabs[i]==1)[0][0] - 1
            if index != -1: 
                if index < 10: final_tab += '---'+str(index)+'--'
                else: final_tab += '--'+str(index)+'--'
                continue
            final_tab += '------'
        final_tab+='\n'
        for i in range(1,len(tabs),6):
            if len(np.nonzero(tabs[i]==1)[0])!=0:
                index = np.nonzero(tabs[i]==1)[0][0] - 1
            if index != -1: 
                if index < 10: final_tab += '---'+str(index)+'--'
                else: final_tab += '--'+str(index)+'--'
                continue
            final_tab += '------'
        final_tab+='\n'
        for i in range(0,len(tabs),6):
            if len(np.nonzero(tabs[i]==1)[0])!=0:
                index = np.nonzero(tabs[i]==1)[0][0] - 1
            if index != -1: 
                if index < 10: final_tab += '---'+str(index)+'--'
                else: final_tab += '--'+str(index)+'--'
                continue
            final_tab += '------'
        final_tab+='\n'
        final_tab+='\n'
        final_tab+='\n'
    print(final_tab)

auto_tab(sys.argv[1])
