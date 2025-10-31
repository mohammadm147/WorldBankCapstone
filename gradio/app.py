import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your model
tokenizer = AutoTokenizer.from_pretrained("mian21/flan-t5-bsae-CUSTOM-TRAINED")
model = AutoModelForSeq2SeqLM.from_pretrained("mian21/flan-t5-bsae-CUSTOM-TRAINED")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input Text", placeholder="Enter your question here..."),
    outputs=gr.Textbox(label="Model Output"),
    title="Fine-tuned FLAN-T5-Small Demo",
    description="Fine-tuned T5 model for World Bank Survey Data"
)

demo.launch()
