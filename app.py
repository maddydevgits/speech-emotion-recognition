import streamlit as st
import librosa
import numpy as np
import pandas as pd

import pickle

encoder=pickle.load(open('encoder.sav','rb'))

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model3.h5")
print("Loaded model from disk")

st.title('Speech Emotion Recognition')

file_st=st.file_uploader('Upload Wav File')

if file_st is not None:
    with open('demo.wav','wb') as f:
        f.write(file_st.getbuffer())
    
    X, sample_rate = librosa.load('demo.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)
    livepredictions = (encoder.inverse_transform((livepreds)))
    st.write(livepredictions)



