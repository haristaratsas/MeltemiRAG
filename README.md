# Meltemi RAG

This Jupyter Notebook demonstrates the process of ingesting data, chunking text, creating a vector store, and querying a language model using the RAG (Retrieval-Augmented Generation) approach. The notebook is designed to work with Greek text and utilizes various libraries and tools to achieve its goals.

## Table of Contents

- Installation
- [Data Ingestion](#data-ingestion)
- [Text Chunking](#text-chunking)
- [Creating a Vector Store](#creating-a-vector-store)
- [Querying the Language Model](#querying-the-language-model)
- Evaluation

## Installation

To run this notebook, you need to install the required libraries. The following command installs the necessary packages:

```sh
!pip install openai pymupdf tqdm chromadb spacy sentence-transformers
!python -m spacy download el_core_news_sm
!python -m spacy download en_core_web_sm
```

## Data Ingestion

The notebook starts by importing the necessary libraries and defining functions to download and preprocess PDF files.

```python
import re
import os
import urllib.request
import uuid
import shutil
import fitz
from tqdm.autonotebook import tqdm

def download_pdf(url, output_path):
    os.makedirs(output_path, exist_ok=True)
    local_pdf = f"{output_path}/{uuid.uuid4().hex}.pdf"
    if url == output_path:
        return
    try:
        urllib.request.urlretrieve(url, local_pdf)
    except ValueError:
        shutil.copy(url, local_pdf)
    return local_pdf

def preprocess(text):
    text = text.replace("-\n", "")  # no word breaks
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.+", ".", text)
    return text

def pdf2text(path, start_page=0, end_page=-1):
    print("Parsing PDF")
    doc = fitz.open(path)
    total_pages = doc.page_count
    print(f"PDF contains {total_pages} pages")
    if end_page <= 0:
        end_page = total_pages

    text_list = []
    for i in tqdm(
        range(start_page, end_page),
        desc=f"Converting PDF to text. Pages: {start_page}-{end_page}",
    ):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)
    doc.close()
    return " ".join(text_list)
```

## Text Chunking

The notebook uses spaCy to split the text into sentences and then merges consecutive sentences into chunks of a specified size.

```python
import spacy

nlp_greek = spacy.load("el_core_news_sm")

def chunk_text(text, spacy_nlp=nlp_greek, chunk_size=128):
    print("Split document into sentences using spacy")
    sentences = [sent.text for sent in nlp_greek(text).sents]
    print(f"Document contains #{len(sentences)} sentences")

    print(f"Merge consecutive sentences up to chunk size = {chunk_size}")
    chunks = [sentences[0]]
    for index in tqdm(range(1, len(sentences)), desc="Chunking text"):
        potential_chunk = "\n".join([chunks[-1], sentences[index]])
        num_tokens_in_chunk = len([token for token in spacy_nlp(potential_chunk)])
        if num_tokens_in_chunk < chunk_size:
            chunks[-1] = potential_chunk
        else:
            chunks.append(sentences[index])

    print(f"Created #{len(chunks)} chunks")
    return chunks
```

## Creating a Vector Store

The notebook creates a vector store using ChromaDB and a multilingual sentence transformer model.

```python
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class MultilingualSentenceTransformer(EmbeddingFunction):
    def __init__(self, model="BAAI/bge-m3"):
        self.model = model
        self.embedder = self.initialize_model(model)

    def initialize_model(self, model):
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(model, device="cuda")
        return embedder

    def __call__(self, sentences: Documents) -> Embeddings:
        return self.embedder.encode(sentences, convert_to_numpy=True).tolist()

def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]

def create_collection(chunks, collection_name="rag_from_scratch", batch_size=12):
    print("Create and index vector store")

    client = chromadb.EphemeralClient()
    embedding_function = MultilingualSentenceTransformer()
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
    )

    id_num = 0
    for batch in tqdm(
        batchify(chunks, n=batch_size),
        desc="Indexing vector database",
        total=len(chunks) // batch_size,
    ):
        texts = batch
        collection.add(
            documents=texts,
            ids=[f"id{idx}" for idx in range(id_num, id_num + len(texts))],
        )
        id_num += len(texts)

    return collection
```

## Querying the Language Model

The notebook defines functions to retrieve relevant passages from the vector store and query a language model using the RAG approach.

```python
import openai
from openai

 import

 OpenAI

MELTEMI_API_KEY = "your_meltemi_api_key"
MELTEMI_BASE_URL = "http://your_meltemi_base_url"
MELTEMI_CLIENT = OpenAI(api_key=MELTEMI_API_KEY, base_url=MELTEMI_BASE_URL)

RAG_SYSTEM_PROMPT = """Instructions: Compose a comprehensive reply to the query, based on the provided passages.
If the search results mention multiple subjects with the same name, create separate answers for each.
In your answers, maintain the appropriate knowledge level for a person that is consuming this material.
You MAY include appropriate extra information for providing clear, simple and comprehensive clarifications,
if the user explicitly asks for it, else you only respond based on the provided passages.
Do not make the responses neither too advanced, nor too simplistic.
If the text does not relate at all to the query, propose more appropriate prompt queries based on the book contents.
Provide your answer in the same language as the user's query."""

PLAIN_SYSTEM_PROMPT = """Instructions: Answer the user's questions to the best of your knowledge."""

def query_llm(query, system_prompt, model="gpt"):
    if "gpt" in model:
        model_name = "gpt-3.5-turbo"
        client = GPT_CLIENT
    else:
        model_name = "meltemi"
        client = MELTEMI_CLIENT
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content, response.usage

def plain_query(query, model="gpt"):
    print("---------------------------------------------------------------")
    print("Plain query without using  RAG")
    print("---------------------------------------------------------------")

    prompt = query
    response, usage = query_llm(prompt, PLAIN_SYSTEM_PROMPT, model=model)
    print()
    print("QUERY:")
    print(prompt)
    print("---------------------------------------------------------------")
    print()
    print("LLM RESPONSE:")
    print(response)
    print("---------------------------------------------------------------")
    print()
    print("TOKEN USAGE:")
    print(usage)
    print("---------------------------------------------------------------")

def rag_query(query, model="gpt", collection=collection_greek):
    print("---------------------------------------------------------------")
    print("Query using  RAG")
    print("---------------------------------------------------------------")

    prompt = format_prompt(query, collection)
    response, usage = query_llm(prompt, RAG_SYSTEM_PROMPT, model=model)
    print()
    print("QUERY:")
    print(prompt)
    print("---------------------------------------------------------------")
    print()
    print("LLM RESPONSE:")
    print(response)
    print("---------------------------------------------------------------")
    print()
    print("TOKEN USAGE:")
    print(usage)
    print("---------------------------------------------------------------")
```

## Evaluation

The notebook includes a function to evaluate a JSON file of questions using the RAGAS library.

```python
import json
from ragas.metrics import answer_relevancy, faithfulness

def evaluate_questions(json_file_path, retriever, generator):
    """Evaluates a JSON file of questions using RAGAS.

    Args:
        json_file_path: The path to the JSON file containing the questions.
        retriever: The retriever to use for retrieving relevant documents.
        generator: The generator to use for generating answers.

    Returns:
        A dictionary containing the evaluation results.
    """

    # Load the questions from the JSON file
    with open(json_file_path, 'r') as f:
        questions = json.load(f)

    results = []

    # Evaluate each question
    for question in tqdm(questions, desc="Evaluating questions"):
        context = retriever.retrieve(question['question'])
        answer = generator.generate(question['question'], context)

        # Calculate metrics
        relevancy = answer_relevancy(question['question'], answer, context)
        faithfulness = faithfulness(question['question'], answer, context)

        results.append({
            "question": question['question'],
            "answer": answer,
            "relevancy": relevancy,
            "faithfulness": faithfulness,
        })

    return results
```

## Usage

To use this notebook, follow these steps:

1. Install the required libraries.
2. Download and preprocess the PDF files.
3. Chunk the text into manageable pieces.
4. Create a vector store using ChromaDB.
5. Query the language model using the RAG approach.
6. Evaluate the results using the provided evaluation function.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
