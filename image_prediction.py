import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import PyPDF2

# Install dependencies:
# pip install pytesseract pillow pypdf2

def extract_text_from_image(image_path):
    # Open the image file using Pillow
    with Image.open(image_path) as img:
        # Convert the image to grayscale
        img = img.convert('L')
        # Use Tesseract to extract text from the image
        text = pytesseract.image_to_string(img)
        return text

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Example usage for image
image_path = 'download.jpeg'
extracted_text = extract_text_from_image(image_path)
print(extracted_text)
# Example usage for PDF
# pdf_path = '100-Inspirational-Quotes-About-Life.pdf'
# extracted_text = extract_text_from_pdf(pdf_path)
# print(extracted_text)