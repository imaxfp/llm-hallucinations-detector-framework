from typing import List
import ollama
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# RUN the next models wiht OLLAMA - https://ollama.com/library
LLM_MODELS = {"llama3.1", "gemma2", "phi3"}

class OllamaChat:
    def __init__(self, models=LLM_MODELS):        
        self.models = models

    def chat_with_ollama(self, model_name, role_prompt=None, msg=None):
        stream = ollama.chat(
            model=model_name,            
            messages=[
                {'role': 'system', 'content': 'You are a question-answer helper. Analyze the question and provide a concise answer.'},
                {'role': 'system', 'content': role_prompt},
                {'role': 'system', 'content': 'PAY ATTENTION YOUR Response have to be about 50 or 60 words.'},
                {'role': 'user', 'content': msg}
            ],
            stream=True
        )
        res = []
        for chunk in stream:
            res.append(chunk['message']['content'])
        return ''.join(res)

    
    def text_to_embedding(seld, model_name, txt):
        embed = ollama.embed(model=model_name, input=txt)
        return embed        
    
    def get_text_response_with_cascade_of_models(self, role_prompt, msgEntities: List):
        with ThreadPoolExecutor() as executor:
            futures = []
            
            # Submit all tasks for all msgEntities and models at once
            for msgEntity in msgEntities:
                res_dir = {}
                for mod_name in self.models:
                    future = executor.submit(self.chat_with_ollama, mod_name, role_prompt, msgEntity.question)
                    futures.append((future, mod_name, role_prompt, msgEntity, res_dir))

            # Collect the results as they are completed
            for future in as_completed([f[0] for f in futures]):
                for f in futures:
                    if f[0] == future:
                        mod_name = f[1]
                        role_prompt = f[2]                        
                        msgEntity = f[3]
                        res_dir = f[4]                                                                        
                        res_dir[mod_name] = future.result()

            # Assign results to each msgEntity
            for _, _, _, msgEntity, res_dir in futures:
                msgEntity.llm_answers = res_dir

        return msgEntities
                     
    def get_embeddings_with_cascade_of_models(self, msgEntities: List):                      
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            for msgEntity in msgEntities:
                embed_dir = {}            
                true_answer_embed_dir = {}
                
                for mod_name, llm_answ in msgEntity.llm_answers.items():
                    # Submit embedding tasks to executor for both llm_answ and true_answer
                    future_embed = executor.submit(self.text_to_embedding, mod_name, llm_answ)
                    future_true_embed = executor.submit(self.text_to_embedding, mod_name, msgEntity.true_answer)

                    futures.append((future_embed, future_true_embed, embed_dir, true_answer_embed_dir))

            # Collect the results as they are completed
            for future_embed, future_true_embed, embed_dir, true_answer_embed_dir in futures:
                res = future_embed.result()
                true_embed = future_true_embed.result()

                embed_dir[res.get('model')] = res.get('embeddings')
                true_answer_embed_dir[true_embed.get('model')] = true_embed.get('embeddings')

            for msgEntity in msgEntities:
                msgEntity.llm_embeddings = embed_dir 
                msgEntity.true_answer_embeddings = true_answer_embed_dir 

        return msgEntities    