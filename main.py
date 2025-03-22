import imaplib
import email
from email.header import decode_header
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Falcon-RW-1B model (lightweight, no sentencepiece required)
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to fetch emails
def fetch_emails(last_x_days=1, max_emails=70):
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    date_threshold = (datetime.now() - timedelta(days=last_x_days)).strftime("%d-%b-%Y")
    status, messages = mail.search(None, f'(SINCE "{date_threshold}")')
    email_ids = messages[0].split()

    if len(email_ids) > max_emails:
        email_ids = email_ids[-max_emails:]

    emails = []
    for email_id in email_ids:
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                emails.append(msg)
    return emails

# Parse subject, sender, body, images
def extract_email_data(email_message):
    email_data = {}

    subject, encoding = decode_header(email_message["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8")
    email_data["subject"] = subject

    sender, encoding = decode_header(email_message.get("From"))[0]
    if isinstance(sender, bytes):
        sender = sender.decode(encoding or "utf-8")
    email_data["sender"] = sender

    email_data["text"] = ""
    for part in email_message.walk():
        if part.get_content_type() == "text/plain":
            try:
                email_data["text"] += part.get_payload(decode=True).decode(errors="ignore")
            except:
                pass

    email_data["images"] = []
    for part in email_message.walk():
        if part.get_content_maintype() == "image":
            image_data = part.get_payload(decode=True)
            image = Image.open(BytesIO(image_data))
            email_data["images"].append({
                "filename": part.get_filename(),
                "size": image.size
            })

    return email_data

# Vectorize email bodies
def vectorize_emails(emails):
    texts = [f"Subject: {email['subject']}\nBody: {email['text']}" for email in emails]
    embeddings = embedding_model.encode(texts)
    return embeddings

# Similarity search
def retrieve_relevant_emails(query, emails, embeddings, top_k=3):
    query_embedding = embedding_model.encode(query)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [emails[i] for i in top_indices]

# Falcon response generator
def generate_response(query, relevant_emails):
    context = "\n\n".join([
        f"Subject: {email['subject']}\nBody: {email['text']}"
        for email in relevant_emails
    ])

    prompt = (
        f"You are an AI assistant that answers questions based on the following emails.\n\n"
        f"{context}\n\n"

        "Here is the question by the user:\n"
        f"User Query: {query}\n\nAnswer:"
        "Answer to the point and in a concise manner in a conversational style - max 5 to 7 lines.\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=250,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output.replace(prompt, "").strip()

# Main loop
def main():
    try:
        last_x_days = int(input("Enter number of days to fetch emails (default 1, max 7): ") or 1)
        last_x_days = max(1, min(last_x_days, 7))
    except ValueError:
        print("Invalid input. Using default of 1 day.")
        last_x_days = 1

    emails = fetch_emails(last_x_days=last_x_days, max_emails=70)
    print(f"Fetched {len(emails)} emails.")
    email_data = [extract_email_data(e) for e in emails]
    embeddings = vectorize_emails(email_data)

    user_query = input("Enter your query: ")
    relevant_emails = retrieve_relevant_emails(user_query, email_data, embeddings, top_k=3)
    response = generate_response(user_query, relevant_emails)

    print("\nGenerated Response:\n")
    print(response)

if __name__ == "__main__":
    main()
