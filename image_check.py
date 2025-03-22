import requests
from PIL import Image
import pytesseract
from io import BytesIO

def extract_text_from_image_url(image_url):
    """
    Extracts text from an image URL using Tesseract OCR.
    
    Args:
        image_url (str): The URL of the image.
    
    Returns:
        str: Extracted text from the image.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        
        # Open the image using PIL from the response content
        image = Image.open(BytesIO(response.content))
        
        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error: {e}"

# Example usage
image_url = rimage_url = r"https://ci3.googleusercontent.com/meips/ADKq_NaW3WklK0EgmPr7JL-4VqkIMsZPhEG453VatGLgCMBCNhw2Q3wN-AVMkCjJQjxQVVcc5vuE03Im3OVq_xN9gQu3fVjNJSZYlEmO6hN_YK9hboWR3SI9wJqNkyN0fy3-EiDeiPI5zdkqwqdnB7clGA=s0-d-e1-ft#https://d30a1bfupaeobl.cloudfront.net/images/R95-W5Z-Z45Z/CNT%20EMAILER%20DELHIsuffix.png"
extracted_text = extract_text_from_image_url(image_url)
print("Extracted Text:", extracted_text)