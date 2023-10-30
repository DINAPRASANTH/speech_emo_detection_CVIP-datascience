 # Speech Emotion Recognition
  This project aims to recognize emotions in speech using machine learning techniques. We use the Toronto Emotional Speech Set (TESS) dataset for training and evaluating our models.
  
 # Dataset
  The dataset is [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) and click it to get datasets.
 # Data Preprocessing
- Loaded the dataset and created a DataFrame.
- Extracted Mel-frequency cepstral coefficients (MFCCs) as audio features.
- One-hot encoded the labels.
- Split the dataset into training and testing sets.
 
# Model Architecture

The neural network model used for this project has the following architecture:
model = Sequential([
    LSTM(123, return_sequences = False, input_shape=(40,1)),
    Dense(64, activation = 'relu'),
    Dropout(0.22),
    Dense(32, activation = 'relu'),
    Dropout(0.12),
    Dense(7, activation = 'softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 123)               61500     
                                                                 
 dense (Dense)               (None, 64)                7936      
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dropout_1 (Dropout)         (None, 32)                0         
                                                                 
 dense_2 (Dense)             (None, 7)                 231       
                                                                 
=================================================================
Total params: 71747 (280.26 KB)
Trainable params: 71747 (280.26 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

# Future Improvements
- Experiment with different model architectures.
- Fine-tune hyperparameters.
- Use transfer learning from pre-trained models.
- Augment the dataset for better model generalization.

# Conclusion
This project demonstrates the process of building a speech emotion recognition model using MFCC features and LSTM neural networks. The model achieved an accuracy of 97.72% on the test data.