'''
The Natural Questions (NQ) dataset is a large-scale question-answering benchmark 
that is primarily used for training and evaluating machine learning models. 
It consists of real-world questions from Google search and their corresponding answers based on Wikipedia pages.
'''
import os
import sys
import pandas as pd
import csv
from typing import List, Dict
import uuid
from src.services.ollama_service import OllamaChat
import time
import logging

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

class NaturalQuestionsParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_entities(self, start_line=0, stop_line=None):
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

        # Collect all unique LLM answer keys and embedding keys (column names)
        for entity in entities:
            all_llm_answer_keys.update(entity.llm_answers.keys())
            all_llm_embedding_keys.update(entity.llm_embeddings.keys())
            all_true_answ_embedding_keys.update(entity.true_answer_embeddings.keys())
            

        # For each entity, create a dictionary row
        for entity in entities:
            row = {
                "uid": entity.uid,
                "question": entity.question,
                "true_answer": entity.true_answer,
            }

            # Fill in the llm_answers for each key, if present
            for key in all_llm_answer_keys:
                row[key] = entity.llm_answers.get(key, "")
            
            # Fill in the llm_embeddings for each key, if present
            for emb_key in all_llm_embedding_keys:
                row[f"embedding_{emb_key}"] = entity.llm_embeddings.get(emb_key, "")
                
            # Fill in the llm_embeddings for each key, if present
            for emb_key in all_llm_embedding_keys:
                row[f"true_answer_embedding_{emb_key}"] = entity.llm_embeddings.get(emb_key, "")                
            
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