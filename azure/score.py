import os, json, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def init():
    global model, tokenizer

    # define model/tokenizer path
    modelDir = os.path.join(os.environ["AZUREML_MODEL_DIR"], "flan-t5-small-label-smooth-balanced")
    model = AutoModelForSeq2SeqLM.from_pretrained(modelDir)
    tokenizer = AutoTokenizer.from_pretrained(modelDir)
    model.eval()

def run(rawData):
    try:
        # load data from parameter and get question
        data = json.loads(rawData)
        inputs = data["inputs"]

        # tokenize input
        encoded = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # generate output
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return json.dumps({"predictions": predictions})

    except Exception as e:
        return json.dumps({"error": str(e)})
