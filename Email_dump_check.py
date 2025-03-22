import imaplib
import email
from email.header import decode_header
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER")

def fetch_emails(last_x_days=7):
    """Fetch emails from the last X days (max 20 days)."""
    # Cap the value of X at 20
    last_x_days = min(last_x_days, 20)

    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    # Calculate the date X days ago
    date_threshold = (datetime.now() - timedelta(days=last_x_days)).strftime("%d-%b-%Y")

    # Search for emails received since the date threshold
    status, messages = mail.search(None, f'(SINCE "{date_threshold}")')
    email_ids = messages[0].split()

    emails = []
    for email_id in email_ids:
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                emails.append(msg)

    return emails

def extract_largest_image(email_message):
    """Extract the largest image from the email."""
    largest_image = None
    max_size = 0

    for part in email_message.walk():
        if part.get_content_maintype() == "multipart":
            continue
        if part.get("Content-Disposition") is None:
            continue

        filename = part.get_filename()
        if filename and any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif"]):
            image_data = part.get_payload(decode=True)
            image = Image.open(BytesIO(image_data))
            image_size = image.size[0] * image.size[1]  # Width * Height

            if image_size > max_size:
                max_size = image_size
                largest_image = image

    return largest_image

def extract_email_data(email_message):
    """Extract email metadata and text content."""
    email_data = {}

    # Extract subject
    subject, encoding = decode_header(email_message["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8")
    email_data["subject"] = subject

    # Extract sender
    sender, encoding = decode_header(email_message.get("From"))[0]
    if isinstance(sender, bytes):
        sender = sender.decode(encoding or "utf-8")
    email_data["sender"] = sender

    # Extract text content
    email_data["text"] = ""
    for part in email_message.walk():
        if part.get_content_type() == "text/plain":
            email_data["text"] += part.get_payload(decode=True).decode()

    return email_data

def save_email_data(email_data, email_id):
    """Save email data to a text file."""
    with open("emails_dump.txt", "a", encoding="utf-8") as f:
        f.write(f"Email ID: {email_id}\n")
        f.write(f"Subject: {email_data['subject']}\n")
        f.write(f"Sender: {email_data['sender']}\n")
        f.write(f"Text Content:\n{email_data['text']}\n")
        f.write("-" * 50 + "\n\n")

def save_image(image, email_id):
    """Save the largest image to a file."""
    if image:
        image_filename = f"email_{email_id}_largest_image.png"
        image.save(image_filename)
        print(f"Saved largest image from email {email_id} to {image_filename}")
    else:
        print(f"No image found in email {email_id}")

def main():
    # Ask the user for the number of days (X)
    try:
        last_x_days = int(input("Enter the number of days to fetch emails (max 20): "))
    except ValueError:
        print("Invalid input. Using default value of 7 days.")
        last_x_days = 7

    # Fetch emails from the last X days
    emails = fetch_emails(last_x_days)

    # Clear the dump file if it exists
    if os.path.exists("emails_dump.txt"):
        os.remove("emails_dump.txt")

    # Process each email
    for i, email_message in enumerate(emails):
        print(f"\nProcessing email {i + 1}...")

        # Extract email data
        email_data = extract_email_data(email_message)

        # Save email data to a text file
        save_email_data(email_data, i + 1)

        # Extract the largest image
        largest_image = extract_largest_image(email_message)

        # Save the largest image
        save_image(largest_image, i + 1)

    print("\nEmail data saved to 'emails_dump.txt'.")

if __name__ == "__main__":
    main()