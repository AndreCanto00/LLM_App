from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

def load_model():
    # Option 1: Using pipeline
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        revision="a4f8f3e"
    )
    return summarizer

def predict(text):
    summarizer = load_model()
    # Generate summary
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Create Gradio interface
with gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter text to summarize..."),
    outputs="text",
    title="Text Summarization",
    description="Enter your text and get a summary using DistilBART-CNN model"
) as interface:
    interface.launch()
