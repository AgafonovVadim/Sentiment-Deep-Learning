# Sentiment Analysis Using LSTM

## Project Description  
This project implements a **Sentiment Analysis Model** using **TensorFlow** to classify movie reviews as positive or negative. The model is designed with simplicity and efficiency in mind, leveraging layers such as `Embedding`, `LSTM`, and `Dense`. By utilizing GPU acceleration, the training process is optimized for faster computation. 

Key features include:  
- Efficient handling of text data using embedding layers.  
- Deep learning architecture with stacked LSTM layers for robust temporal dependency representation.  
- Adaptive learning rate control with `ReduceLROnPlateau` for optimized training.  
- Real-world evaluation using metrics like **Accuracy**, **Precision**, **Recall**, and **F1-score**.

---

## Features  
1. **Two-Layer LSTM Architecture**:  
   - Enhanced temporal representation by stacking LSTMs.  
   - Dropout for regularization to reduce overfitting.  

2. **Dynamic Learning Rate Control**:  
   - `ReduceLROnPlateau` to adjust learning rate when validation loss stagnates.  

3. **GPU Acceleration**:  
   - Leverages TensorFlow’s GPU support for faster training.  

4. **Comprehensive Metrics**:  
   - Evaluates model performance with Accuracy, Precision, Recall, and F1-score.  

5. **Scalable Dataset Support**:  
   - Easily handles datasets with thousands of labeled reviews.

---

## Dataset  
The dataset consists of two CSV files:  
- **positive.csv**: Contains 7000 positive movie reviews.  
- **negative.csv**: Contains 7000 negative movie reviews.  

Each row represents a single movie review in text form.

---

## Requirements  

- Python 3.7+  
- TensorFlow 2.7+  
- NumPy  
- Pandas  
- Scikit-learn  

---

## Usage  

### **1. Prepare the Dataset**  
Place the `positive.csv` and `negative.csv` files in the `dataset/` directory.  

### **2. Train the Model**  
Run the training file step-by-step:  
```bash
python predict.py
```

### **3. Evaluate the Model**  
The script outputs the following metrics on the test set:  
- Accuracy  
- Precision  
- Recall  
- F1-score  

Example output:  
```plaintext
Accuracy: 0.9214
Precision: 0.9158
Recall: 0.9300
F1-score: 0.9229
```

### **4. Save the Model**  
You could save the trained model as `sentiment_model.h5` for future use.  


---

## Model Architecture  

- **Embedding Layer**: Converts text into dense vectors.  
- **LSTM Layers**: Captures temporal dependencies in text.  
- **Dense Layer**: Outputs the binary classification result.  

---

## Key Results  

Training over 100 epochs resulted in the following metrics:  

| Metric       | Score    |  
|--------------|----------|  
| Accuracy     | 0.9214   |  
| Precision    | 0.9158   |  
| Recall       | 0.9300   |  
| F1-score     | 0.9229   |  

---

## Future Improvements  

1. **Use Pretrained Embeddings**: Incorporate GloVe or FastText embeddings for better text representation.  
2. **Data Augmentation**: Introduce techniques like back translation to expand the dataset.  
3. **Transformer Architecture**: Explore models like BERT or GPT for more sophisticated text processing.  
4. **Class Imbalance Handling**: If class distribution is unequal, apply SMOTE or weighted loss functions.  

---

## License
This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

## Acknowledgments  

Special thanks to **TensorFlow** and **Scikit-learn** for providing robust libraries to streamline model development and evaluation.  
