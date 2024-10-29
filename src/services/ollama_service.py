import csv
from typing import List
import uuid
import ollama
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import time
import logging

import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RUN the next models wiht OLLAMA - https://ollama.com/library
#LLM_MODELS = {"llama3.1", "gemma2", "phi3"}
LLM_MODELS = {"llama3.1"}


class QuestionEntity:
    def __init__(self, uid, question, true_answer):
        self.uid = uid
        self.question = question
        self.true_answer = true_answer
        self.llm_answers = {}
        self.llm_embeddings = {}
        self.true_answer_embeddings = {}

    def __repr__(self):        
        return f"QuestionEntity(uid='{self.uid}', question='{self.question}', true_answer='{self.true_answer}')"
    
    def truncate(self, text, length=80):
        """Truncate text to the specified length with '...' if needed."""
        return text[:length] + '...' if len(text) > length else text

    def __str__(self):
        return (
            f"UID: {self.uid}\n"
            f"Question: {self.truncate(self.question)}\n"
            f"True Answer: {self.truncate(self.true_answer)}\n"
            f"True Answer Embeddings: {self.truncate(str(self.true_answer_embeddings))}\n"
            f"LLM Answers: {self.truncate(str(self.llm_answers))}\n"
            f"LLM Embeddings: {self.truncate(str(self.llm_embeddings))}\n"
        )

class OllamaChat:
    def __init__(self, model_names=LLM_MODELS):        
        self.model_names = model_names

    def chat_with_ollama_with_prompt(self, model_name, role_prompt=None, msg=None):
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
    
    def text_to_embedding(self, model_name, txt):
        embed = ollama.embed(model=model_name, input=txt)
        return embed        
    
    def get_text_response_with_cascade_of_models(self, role_prompt, msgEntities: List):
        with ThreadPoolExecutor() as executor:
            futures = []
            
            # Submit all tasks for all msgEntities and models at once
            for msgEntity in msgEntities:
                res_dir = {}
                for mod_name in self.model_names:
                    future = executor.submit(self.chat_with_ollama_with_prompt, mod_name, role_prompt, msgEntity.question)
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
                     
    def get_true_answer_embeddings_with_cascade_of_models(self, msgEntities: List[QuestionEntity]):                                          
            for msgEntity in msgEntities:                                                        
                # Submit embedding tasks to executor for both llm_answ and true_answer                                        
                logging.debug(f"True Answer text to embedding => {str(msgEntity.true_answer)[:20]}")                                          

                #Convert true answer from the dataset to the embedding by specific model  
                for llm_model_name in self.model_names:  
                    true_answer = self.text_to_embedding(llm_model_name, msgEntity.true_answer)
                    true_answer_embedding = true_answer.get("embeddings")[0]
                    logging.debug(f"True Answer embedding => {str(true_answer_embedding)[:20]}")
                    msgEntity.true_answer_embeddings[llm_model_name] = true_answer_embedding
            return msgEntities         
    
    def get_personal_llm_answer_embeddings_with_cascade_of_models(self, msgEntities: List):                                          
            for msgEntity in msgEntities:
                for llm_model_name, llm_answ in msgEntity.llm_answers.items():
                    # Submit embedding tasks to executor for both llm_answ and true_answer
                    logging.debug(f"Get embedding from {llm_model_name}")
                    logging.debug(f"LLM answer text to embedding => {llm_answ}")
                    logging.debug(f"True Answer text to embedding => {msgEntity.true_answer}")
                                        
                    #Convert answer frome the specifict model to the embeding by specifci model
                    llm_answer = self.text_to_embedding(llm_model_name, llm_answ)
                    embedding = llm_answer.get("embeddings")[0]
                    logging.debug(f"LLM answer embedding => {str(embedding)[:20]}")
                    msgEntity.llm_embeddings[llm_model_name] = embedding
            return msgEntities 

    #ollama responses generator 
    def log_time_taken(self, step_name, start_time):
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            logging.info(f"Time taken for {step_name}: {minutes} minutes and {seconds:.2f} seconds")
        else:
            logging.info(f"Time taken for {step_name}: {elapsed_time:.2f} seconds")    

    #TODO improve the code DRY
    def true_answers_to_embeddings_save_response(self, file_path: str, destination_file: str,  start_line, stop_line):
        # Start measuring time
        total_start_time = time.time()

        #Parse initial file
        parser = NaturalQuestionsParser(file_path)
        logging.info("Read entities from file")
        step_start_time = time.time()
        qaEntities = parser.parse_entities(start_line=start_line, stop_line=stop_line)
        self.log_time_taken("parsing entities", step_start_time)
        
        #Create embeddings with Ollama and add them to the dataset
        qaEntities = self.get_true_answer_embeddings_with_cascade_of_models(qaEntities)    

        #Save results
        logging.info("Store results")
        step_start_time = time.time()
        parser.store_llms_responses(qaEntities, destination_file)
        self.log_time_taken("storing results", step_start_time)

        # Total time
        self.log_time_taken("total execution", total_start_time)            

    #TODO improve the code DRY        
    def chat_ollama_get_llm_answers_with_embedding_save_response(self, file_path: str, destination_file: str, role_prompt, start_line, stop_line):
    
        total_start_time = time.time()
        parser = NaturalQuestionsParser(file_path)
        logging.info("Read entities from file")
        step_start_time = time.time()
        qaEntities = parser.parse_entities(start_line=start_line, stop_line=stop_line)
        self.log_time_taken("parsing entities", step_start_time)
                
        logging.info("Chat with Ollama, get responses from the models")
        step_start_time = time.time()
        msgEntities = self.get_text_response_with_cascade_of_models(role_prompt, qaEntities)
        self.log_time_taken("chatting with models", step_start_time)
        
        # Send questions to the LLM models and collect responses from the LLMs    
        logging.info("Answers to embeddings")
        step_start_time = time.time()
        msgEntities = self.get_personal_llm_answer_embeddings_with_cascade_of_models(qaEntities)
        self.log_time_taken("chatting with models", step_start_time)

        # 4. Save results
        logging.info("Store results")
        step_start_time = time.time()
        parser.store_llms_responses(msgEntities, destination_file)
        self.log_time_taken("storing results", step_start_time)

        # Total time
        self.log_time_taken("total execution", total_start_time)




class NaturalQuestionsParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_entities(self, start_line=0, stop_line=None) -> List[QuestionEntity]:
        """
        Parse the CSV file and return a list of QuestionEntity objects starting from a specific line
        and stopping before a specific line.

        :param start_line: The line number to start reading from (0-indexed). Default is 0.
        :param stop_line: The line number to stop reading (exclusive). If None, read until the end of the file.
        :return: List of QuestionEntity objects.
        """
        entities = []
        with open(self.file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            # Skip rows until the start_line
            for _ in range(start_line):
                next(reader, None)

            for i, row in enumerate(reader, start=start_line):
                if stop_line is not None and i >= stop_line:
                    break
                entity = QuestionEntity(
                    uid=row['uid'],
                    question=row['question'],
                    true_answer=row['true_answer'],                    
                )
                entities.append(entity)
        return entities

    
    def delete_empty_or_very_long_short_answs(self):
        """
        Updates the dataset by removing rows where `true_answer` is empty, has fewer than 10 words, or more than 80 words.
        """
        rows_to_keep = []
        
        # Read the file and filter rows based on the conditions
        with open(self.file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames  # Keep field names for rewriting
            for row in reader:
                true_answer = row['true_answer']
                word_count = len(true_answer.split())
                
                # Check if true_answer is non-empty and has between 10 and 80 words
                if true_answer and 10 <= word_count <= 80:
                    rows_to_keep.append(row)

        # Overwrite the original file with filtered rows
        with open(self.file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)
            
    def add_uid_if_empty(self):
        df = pd.read_csv(self.file_path)

        # Function to generate UID
        def generate_uid():
            return str(uuid.uuid4())

        # Apply UID generation where 'UID' column is NaN or empty
        df['uid'] = df['uid'].apply(lambda x: generate_uid() if pd.isna(x) or x == '' else x)

        # Save the updated dataframe back to CSV
        df.to_csv(self.file_path, index=False)

        return df


    def store_llms_responses(self, entities: list, res_file_path: str):
        # Prepare a list of dictionaries for storing rows
        data = []
        all_llm_answer_keys = set()
        all_llm_embedding_keys = set()
        all_true_answ_embedding_keys = set()

 
        # For each entity, create a dictionary row
        for entity in entities:
            all_llm_answer_keys.update(entity.llm_answers.keys())
            all_llm_embedding_keys.update(entity.llm_embeddings.keys())
            all_true_answ_embedding_keys.update(entity.true_answer_embeddings.keys())

            row = {
                "uid": entity.uid,
                "question": entity.question,
                "true_answer": entity.true_answer,
            }

            # Fill in the llm_answers for each key, if present
            for key in all_llm_answer_keys:
                row[key] = entity.llm_answers.get(key, "")
            
            for emb_key in all_llm_embedding_keys:
                row[f"embedding_{emb_key}"] = entity.llm_embeddings.get(emb_key, "")
                
            for emb_key in all_true_answ_embedding_keys:
                row[f"true_answer_embedding_{emb_key}"] = entity.true_answer_embeddings.get(emb_key, "")                
            
            data.append(row)

        # Convert the list of dictionaries to a DataFrame
        new_df = pd.DataFrame(data)

        # Check if the CSV file already exists
        if os.path.exists(res_file_path):
            # Read the existing CSV file
            existing_df = pd.read_csv(res_file_path)

            # Filter out entities that already exist (based on 'uid')
            new_df = new_df[~new_df['uid'].isin(existing_df['uid'])]

            # Append the new records to the existing dataframe
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # If no file exists, just use the new dataframe
            combined_df = new_df

        # Save the updated dataframe to the CSV file
        combined_df.to_csv(res_file_path, index=False)

        return combined_df
    
    def read_llm_results_from_csv(self):
        entities = []

        with open(self.file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                # Extract uid, question, and true_answer
                uid = row['uid']
                question = row['question']
                true_answer = row['true_answer']

                # Create QuestionEntity instance
                entity = QuestionEntity(uid, question, true_answer)

                # Extract LLM answers, embeddings, and true answer embeddings
                for key, value in row.items():
                    if key not in ['uid', 'question', 'true_answer']:
                        if key.startswith('embedding_'):
                            # Store in llm_embeddings by removing the 'embedding_' prefix
                            emb_key = key[len('embedding_'):]
                            entity.llm_embeddings[emb_key] = value
                        elif key.startswith('true_answer_embedding_'):
                            # Store in true_answer_embeddings by removing the 'true_answer_embedding_' prefix
                            emb_key = key[len('true_answer_embedding_'):]
                            entity.true_answer_embeddings[emb_key] = value
                        else:
                            # Store in llm_answers
                            entity.llm_answers[key] = value

                # Add entity to the list
                entities.append(entity)

        return entities            