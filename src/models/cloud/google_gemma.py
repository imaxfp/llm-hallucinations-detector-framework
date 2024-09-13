import torch
from transformers import pipeline

class GoogleGemma:
    def __init__(self, model_name="google/gemma-2-2b", device="cuda"):
        # Initialize the pipeline in the constructor
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device  # replace with "mps" to run on a Mac device
        )

    def generate_text_response(self, message):
        # Set up the initial text input
        messages = [
            {"role": "system", "content": "You are a question-answering helper, designed to assist users with their questions by providing accurate and helpful responses"},
            {"role": "user", "content": message}
        ]

        # Generate the response using the pipeline
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
        )
        
        # Return the generated text from the response
        return outputs[0]["generated_text"]

if __name__ == "__main__":
    print(GoogleGemma().generate_text_response("Who is Socrates?"))
