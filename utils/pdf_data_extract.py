import os
from PIL import Image
import pytesseract
import json

# ✅ Set the path to your Tesseract installation
# Make sure this path matches your actual installation path
pytesseract.pytesseract.tesseract_cmd = r"D:\tesrect\tesseract.exe"

# ✅ Folder containing your images
folder_path = "./data/pdf_images"

# ✅ List to store extracted text for each image
data = []

# ✅ Supported image extensions
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# ✅ Loop through all image files in the folder
for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith(valid_extensions):
        image_path = os.path.join(folder_path, filename)

        try:
            # Open and extract text from the image
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(
                    img, lang="eng"
                )  # you can change lang if needed

            # Store the results
            data.append({"path": image_path, "text": text.strip()})

            print(f"✅ Extracted text from: {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# ✅ Print the collected data neatly
print("\n=== Extracted Data ===")
for item in data:
    print(json.dumps(item, indent=2, ensure_ascii=False))

# (Optional) ✅ Save to a JSON file
output_file = "./data/extracted_pdf_images_text.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n💾 All extracted text saved to: {output_file}")
