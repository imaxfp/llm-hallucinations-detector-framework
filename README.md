TODO add description 

#### Setup env:

- python3 -m venv ./venv
- source ./venv/bin/activate
- pip3 install -r requirements.txt


```
Make sure to update your transformers installation via pip install --upgrade transformers.
```


### Run your LLMs localy in ofline mode
Ollama - Running LLMs locally, putting the control directly in your hands.
You can enjoy the benefits of LLMs while maintaining data privacy and control over your computational resources.

https://writingmate.ai/blog/guide-to-running-llama-2-locally
https://ollama.com/library

### Setup Models Offline 

1. download and install https://ollama.com/download
2. run in your faworite terminal - ollama run llama3.1
3. run in your faworite terminal - ollama run gemma2
4. run simultaneously several models

```

 ~ ollama run llama3.1
pulling manifest
pulling 8eeb52dfb3bb... 100% ▕████████████████████████████████████████████████████████▏ 4.7 GB
pulling 73b313b5552d... 100% ▕████████████████████████████████████████████████████████▏ 1.4 KB
pulling 0ba8f0e314b4... 100% ▕████████████████████████████████████████████████████████▏  12 KB
pulling 56bb8bd477a5... 100% ▕████████████████████████████████████████████████████████▏   96 B
pulling 1a4c3c319823... 100% ▕████████████████████████████████████████████████████████▏  485 B
verifying sha256 digest
writing manifest
success
```

#### Dataset
Current dataset:
https://www.kaggle.com/datasets/frankossai/natural-questions-dataset

Useful datasets:
https://ai.google.com/research/NaturalQuestions/download
https://www.kaggle.com/datasets/memocan/data-science-interview-q-and-a-treasury




#####  Why is better to run LLMs locally? 

- **Uninterrupted access:** You won't have to worry about rate limits, downtime, and unexpected service disruptions.

- **Improved performance:** The response generation is fast without lag or latencies. Even on mid-level laptops, you get speeds of around 50 tokens per second.

- **Enhanced security:** You have full control over the inputs used to fine-tune the model, and the data stays locally on your device.

- **Reduced costs:** Instead of paying high fees to access the APIs or subscribe to the online chatbot, you can use Llama 3 for free.

- **Customization and flexibility:** You can customize models using hyperparameters, add stop tokes, and change advanced settings.

- **Offline capabilities:** Once you have downloaded the model, you don't need an internet connection to use it.

- **Ownership:** You have complete ownership and control over the model, its data, and its outputs.
 
