import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()
    return text_data

def preprocess_data(text_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text_data])
    total_words = len(tokenizer.word_index) + 1

    sequences = []
    for line in text_data.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in sequences])
    sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

    X = sequences[:,:-1]
    y = sequences[:,-1]
    return X, y, total_words, max_sequence_len, tokenizer

def create_model(total_words, max_sequence_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 100, input_shape=(max_sequence_len-1,)),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(total_words, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, X, y, epochs=100):
    model.fit(X, y, epochs=epochs, verbose=1)
    return model

def generate_text(model, tokenizer, seed_text, max_sequence_len, next_words=50):
    generated_text = [seed_text]

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        generated_text.append(output_word)

    generated_text = ' '.join(generated_text)
    return generated_text

def main():
    file_path = 'poem.txt'
    text_data = load_text_data(file_path)

    X, y, total_words, max_sequence_len, tokenizer = preprocess_data(text_data)

    model = create_model(total_words, max_sequence_len)
    model = train_model(model, X, y)

    seed_text = "Once upon a time,"
    generated_story = generate_text(model, tokenizer, seed_text, max_sequence_len, next_words=150)
    print(generated_story)

if __name__ == "__main__":
    main()
