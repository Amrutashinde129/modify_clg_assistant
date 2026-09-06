import fitz


def extract_text_from_pdf(pdf_file):
    text = ""

    document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in document:
        text += page.get_text()

    document.close()

    return text