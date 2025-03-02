# Phi-2 Finetuning model demo

## Introduction

As part of The School of AI's ERA-V3 course's Assignment 18- I have fine-tuned an open-source phi-2 model by Microsoft on the OASST-1 (Open Assistant) dataset.

This repository contains a Gradio-based web application for generating text using a fine-tuned Phi-2 model with PEFT adapters. The application allows users to input prompts and generate text based on those prompts. It also includes sample questions that users can click to quickly generate text.

## Sources

- [Microsoft's Phi-2 model](https://huggingface.co/microsoft/phi-2)
- [OpenAssistant OASST-1 dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)

## Features

- Generate text using a fine-tuned Phi-2 model.
- Input custom prompts and adjust generation parameters.
- Click sample questions to quickly generate text.
- Uses PEFT adapters for efficient fine-tuning.

## Requirements

- Python 3.10 or higher
- The following Python packages (specified in `requirements.txt`):
  - `huggingface_hub`
  - `transformers>=4.45.1`
  - `peft`
  - `accelerate`
  - `torch`
  - `gradio`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shettysaish20/era-phi-2-oasst-instruct.git
   cd phi-2-oasst-instruct
   ```

2. Create and activate a virtual environment (optional but recommended):
    ```Python
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```Python
    pip install -r requirements.txt
    ```

4. Run the application
    ```bash
    python app.py
    ```

5. Open your web browser and go to http://localhost:7860 to access the Gradio interface.

## Loading using Docker

To deploy the application using Docker, follow these steps:
1. Build the Docker image:
    ```bash
    docker build -t era-phi-2-oasst-instruct .
    ```

2. Run the Docker container:
    ```bash
    docker run -p 7860:7860 era-phi-2-oasst-instruct
    ```

3. Open your web browser and go to http://localhost:7860 to access the Gradio interface.  

## Fine-tuning logs

Trained for 500 steps with checkpointing after every 100 steps

![Training logs chart](images\training_logs.png)
