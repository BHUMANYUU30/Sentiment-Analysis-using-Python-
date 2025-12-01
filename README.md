# 🧠 Sentiment Analysis using Python

A Deep Learning–based text classification project that predicts whether a given text expresses **positive** or **negative** sentiment using LSTM and NLP preprocessing.

---

## 📌 Overview

Sentiment analysis is widely used across industries like retail, telecom, banking, and customer support. It helps businesses understand customer opinions, feedback, and emotions toward their services.

This repository implements a **binary sentiment classifier** using Python, NLP, and an LSTM-based deep learning model. The model processes tweets, converts them into embeddings, learns patterns, and classifies sentiment accurately.

---

## 📊 Dataset

The dataset contains **14,000+ tweet samples**, each labeled as:

* positive
* negative
* neutral

For this binary classification task, **neutral tweets are removed**.

You can download the dataset here:
📥 *Project Dataset* (add your dataset link here)

---

## 🛠️ Tech Stack & Libraries

* **Python 3.x**
* **Pandas 1.2.4**
* **NumPy**
* **Matplotlib 3.3.4**
* **TensorFlow 2.4.1**
* **Keras Tokenizer & LSTM Layers**

Install dependencies:

```bash
pip install pandas matplotlib tensorflow
```

---

## 📂 Project Structure

```
📦 Sentiment-Analysis
├── dataset/
│   └── Tweets.csv
├── sentiment_analysis.ipynb
├── model/
│   └── trained_model.h5    (optional)
├── README.md
└── requirements.txt
```

---

## 🧹 Data Preprocessing

Steps involved in preparing the data:

1. Load the dataset
2. Select `text` and `sentiment` columns
3. Remove neutral tweets
4. Convert labels to numeric values
5. Tokenize text using Keras `Tokenizer`
6. Convert sentences into sequences
7. Apply padding to equalize sentence length

Example:

```python
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)
```

---

## 🧠 Model Architecture

The LSTM-based deep learning model includes:

* Embedding Layer
* SpatialDropout1D (to prevent overfitting)
* LSTM Layer
* Dropout Layer
* Dense Output Layer (Sigmoid activation)

Model example:

```python
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## 🏋️ Training

Train settings used:

* Epochs: 5
* Batch size: 32
* Validation split: 20%

Training code:

```python
history = model.fit(
    padded_sequence,
    sentiment_label[0],
    validation_split=0.2,
    epochs=5,
    batch_size=32
)
```

Typical results:

* ✔️ **~96% training accuracy**
* ✔️ **~94% validation accuracy**

(Results may vary depending on preprocessing and random seed.)

---

## 📈 Performance Visualization

Plot accuracy:

```python
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
```

Plot loss:

```python
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
```

---

## 🔮 Sentiment Prediction

Prediction function:

```python
def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted Sentiment:", sentiment_label[1][prediction])
```

Example:

```python
predict_sentiment("I enjoyed my journey on this flight.")
# Output: positive

predict_sentiment("This is the worst flight experience ever!")
# Output: negative
```

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/sentiment-analysis-python.git
cd sentiment-analysis-python
```

2. Install required libraries

```bash
pip install -r requirements.txt
```

3. Open the notebook

```bash
jupyter notebook sentiment_analysis.ipynb
```

4. Run all cells to train and test the model.

---

## 📝 Results

* Achieved **94%+ validation accuracy** in experiments.
* Works well on short and long text inputs.
* Easily extendable to multi-class classification or to use pre-trained embeddings (e.g., GloVe, Word2Vec) or transformer-based models for improved performance.

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or pull requests to improve the project. Please follow standard contribution guidelines (issue templates, PR descriptions, tests if applicable).

---

## 📬 Contact

If you have questions or suggestions, reach out:
📧 **[bhumanyunbharty@gmail.com](mailto:bhumanyunbharty@gmail.com)**
