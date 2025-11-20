import fitz # PyMuPDF
from PIL import Image
import io

# def qrant

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """
    Extracts images from a PDF file and saves them to a specified folder.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_folder (str): The folder where extracted images will be saved.
    """
    doc = fitz.open(pdf_path)
    image_count = 0

    # Create output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)

            if base_image:
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Create a Pillow Image object
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    output_filename = os.path.join(output_folder, f"image_page{page_num+1}_{img_index+1}.{image_ext}")
                    image.save(output_filename)
                    image_count += 1
                except Exception as e:
                    print(f"Could not save image {xref} on page {page_num+1}: {e}")

    doc.close()
    print(f"Extracted {image_count} images to '{output_folder}'")

# Example usage:
# Replace 'your_document.pdf' with the actual path to your PDF file
extract_images_from_pdf('./data/F18-ABCD-000.pdf', './data/pdf_images2')