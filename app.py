from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Enable CORS so frontend (other port) can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in dev allow all; later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + tokenizer
model = tf.keras.models.load_model("bilstm_model.keras")
with open("tokanizer_dl.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 200

@app.post("/predict")
def predict(data: dict):
    text = data["text"]
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")

    prob = float(model.predict(padded)[0][0])
    prediction = "Fake" if prob > 0.5 else "Real"

    return {"prediction": prediction, "probability": prob}
