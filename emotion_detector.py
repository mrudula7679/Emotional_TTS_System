from transformers import pipeline
from docx import Document
import PyPDF2
from PIL import Image
import pytesseract

# Set Tesseract path (Windows users only)
# If Tesseract is installed somewhere else, update this path.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        try:
            _classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=False
            )
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            raise e
    return _classifier



EMOTION_MAP = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "neutral"
}



def detect_emotion(text: str) -> str:
    text = text.strip()
    if not text:
        return "neutral"
    classifier = get_classifier()
    try:
        result = classifier(text[:512])  # limit input length
        label = result[0]["label"]
        return EMOTION_MAP.get(label.lower(), "neutral")
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "neutral"



def read_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return ""



def read_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return ""



def read_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text



def read_image(file_path: str) -> str:
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error reading Image file: {e}")
        return ""



if __name__ == "__main__":
    print("Select input type:")
    print("1. Manual input")
    print("2. Text file (.txt)")
    print("3. Word document (.docx)")
    print("4. PDF (.pdf)")
    print("5. Image file (.png, .jpg, .jpeg)")
    
    choice = input("Enter choice [1-5]: ")

    if choice == "1":
        text = input("Enter your text: ")
    elif choice == "2":
        path = input("Enter path to .txt file: ")
        text = read_txt(path)
    elif choice == "3":
        path = input("Enter path to .docx file: ")
        text = read_docx(path)
    elif choice == "4":
        path = input("Enter path to .pdf file: ")
        text = read_pdf(path)
    elif choice == "5":
        path = input("Enter path to image file: ")
        text = read_image(path)
    else:
        print("Invalid choice!")
        exit(0)

    emotion = detect_emotion(text)
    print("\nDetected emotion:", emotion)
