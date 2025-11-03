import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load model
tokenizer = AutoTokenizer.from_pretrained("mian21/flan-t5-small-label-smooth-balanced")
model = AutoModelForSeq2SeqLM.from_pretrained("mian21/flan-t5-small-label-smooth-balanced")

def predict(text):
    if not text.strip():
        return "Please enter your question."
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# css for styling
customCss = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 850px !important;
    margin: auto;
}

#header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 30px;
}

#header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 700;
}

#header p {
    margin: 10px 0 0 0;
    font-size: 1.1em;
    opacity: 0.95;
}

#main-box {
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.footer {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    color: #666;
    font-size: 0.9em;
}

.examples-box {
    margin-top: 20px;
}
"""

# create the blocks interface
with gr.Blocks(css=customCss, theme=gr.themes.Soft()) as demo:

    # header
    with gr.Column(elem_id="header"):
        gr.Markdown("# FLAN-T5 Demo")
        gr.Markdown("Fine-tuned model for World Bank Survey Data")

    # main content
    with gr.Column(elem_id="main-box"):
        gr.Markdown("### Ask your question about the survey data")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Question",
                    placeholder="e.g., What body/agency grants banking licenses in the United States?",
                    lines=4,
                    show_label=True
                )

                with gr.Row():
                    clear_btn = gr.ClearButton(components=[input_text])
                    submit_btn = gr.Button("Generate Answer", variant="primary", scale=2)

        output_text = gr.Textbox(
            label="Model Response",
            lines=6,
            show_label=True,
            interactive=False
        )

        # examples section
        with gr.Accordion("Example Questions", open=False, elem_classes="examples-box"):
            gr.Examples(
                examples=[
                    ["What is the minimum capital requirement in France?"],
                    ["Who regulates banks in Japan?"],
                ],
                inputs=input_text,
            )

    # footer
    gr.Markdown(
        """
        <div class="footer">
        Built with Transformers and Gradio | Model: FLAN-T5-Small fine tuned on World Bank Survey Data
        </div>
        """,
        elem_classes="footer"
    )

    # event handlers
    submit_btn.click(fn=predict, inputs=input_text, outputs=output_text)
    input_text.submit(fn=predict, inputs=input_text, outputs=output_text)

demo.launch()
