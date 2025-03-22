import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Falcon 1B model (small + no sentencepiece needed)
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Sample email content
sample_text = """
The MS Admissions Ambassadors invite you to a Technical Interview Workshop by the NYC Interview Tech Squad.

Date: Friday, April 4th  
Time: 5:30 - 7:00 PM EST  
Location: 7th Floor, CDS

Topics:  
- How to ace technical interviews  
- Live Leetcode problem solving  
- Network with engineers and students  
"""

test_emails = [
    {"subject": "Project Update", "sender": "john@example.com", "text": sample_text},
    {"subject": "Budget Approval", "sender": "jane@example.com", "text": sample_text},
    {"subject": "Meeting Schedule", "sender": "alice@example.com", "text": "Let's schedule a meeting to discuss next steps."}
]

def vectorize_emails(emails):
    texts = [f"Subject: {email['subject']}\nBody: {email['text']}" for email in emails]
    return embedding_model.encode(texts)

def retrieve_relevant_emails(query, emails, embeddings, top_k=2):
    query_embedding = embedding_model.encode(query)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [emails[i] for i in top_indices]

def generate_response(query, relevant_emails):
    context = "\n\n".join([f"Subject: {e['subject']}\nBody: {e['text']}" for e in relevant_emails])
    prompt = (
        f"You are an AI assistant that answers questions based on the email content.\n\n"
        f"{context}\n\n"
        f"User Query: {query}\n\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.replace(prompt, "").strip()

def main():
    embeddings = vectorize_emails(test_emails)
    user_query = input("Enter your query: ")
    relevant_emails = retrieve_relevant_emails(user_query, test_emails, embeddings)
    response = generate_response(user_query, relevant_emails)
    print("\nGenerated Response:\n" + response)

if __name__ == "__main__":
    main()
