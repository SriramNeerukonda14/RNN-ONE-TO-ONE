import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------
# Create model folder
# ---------------------------------------
os.makedirs("model", exist_ok=True)

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("IMDB_Dataset_CLEANED.csv")

df = df[['review', 'sentiment']]
df.dropna(inplace=True)

# ---------------------------------------
# Encode Labels
# negative = 0
# positive = 1
# ---------------------------------------
encoder = LabelEncoder()
df["sentiment"] = encoder.fit_transform(df["sentiment"])

# ---------------------------------------
# Tokenizer
# ---------------------------------------
VOCAB_SIZE = 10000
MAX_LENGTH = 200

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df["review"])

sequences = tokenizer.texts_to_sequences(df["review"])

X = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

y = df["sentiment"]

# ---------------------------------------
# Train Test Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# Build Model
# ---------------------------------------
model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=64,
        input_length=MAX_LENGTH
    ),

    SimpleRNN(64),

    Dense(1, activation="sigmoid")
])

# ---------------------------------------
# Compile
# ---------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------
# Train
# ---------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64,
    callbacks=[early_stop]
)

# ---------------------------------------
# Evaluate
# ---------------------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")

# ---------------------------------------
# Save Model
# ---------------------------------------
model.save("model/sentiment_rnn.keras")

# ---------------------------------------
# Save Tokenizer
# ---------------------------------------
with open("model/tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

# ---------------------------------------
# Save Max Length
# ---------------------------------------
with open("model/maxlen.pkl","wb") as f:
    pickle.dump(MAX_LENGTH,f)

print("\nEverything Saved Successfully!")