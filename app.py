import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer # BitsAndBytesConfig
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA for Gradio
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_HOME"] = "/tmp/huggingface_home"


# Model and tokenizer loading
model_name = "microsoft/phi-2"  # Replace with your base model name
adapter_path = "./adapter"  # Path to your adapter directory (relative to app.py)

# temporarily commented quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    # quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # Use float16 for faster inference
)
model = PeftModel.from_pretrained(model, adapter_path)
model.to(torch.device("cpu"))  # Explicitly move adapter to CPU
model.eval()

# Inference function
def generate_text(prompt, max_length=200, temperature=0.7, top_p=0.9):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(torch.device("cpu"))
    attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(torch.device("cpu"))

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Sample questions
sample_questions = [
    "Write a short story about a dog who becomes a detective.",
    "What is 2+2?",
    "Write a Flask App in python to say 'Hello World!'",
    "Give me a short 200-word essay on 'monospony'.",
]

# Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=5, label="Prompt"),
        gr.Slider(minimum=50, maximum=500, value=250, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top P"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Phi-2 OASST Fine-Tuning Demo",
    description="Generate text using a fine-tuned Phi-2 model with PEFT adapters. Click a sample question below to get started!",
    examples=[[q, 250, 0.1, 0.9] for q in sample_questions],  # Add examples
)

iface.launch()