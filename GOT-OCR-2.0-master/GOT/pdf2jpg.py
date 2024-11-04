def convert_pdf_to_jpg(inputpdf_path, outputjpg_folder):
    import os
    os.makedirs(outputjpg_folder, exist_ok=True)
    images = convert_from_path(inputpdf_path)
    for i, image in enumerate(images):
        image_path = os.path.join(outputjpg_folder, f'{i+1:05}.jpg')
        image.save(image_path, 'JPEG')
        # print(f'Saved: {image_path}')

import os
from pdf2image import convert_from_path

pdf_folder = 'pdfs'
jpg_folder = 'images'

# Create the output folder if it doesn't exist
pdf_files = os.listdir(pdf_folder)

# iterate the files
for pdf_file in pdf_files:
    lhh_name = os.path.splitext(pdf_file)[0]

    lhh_pdf_path = os.path.join(pdf_folder, pdf_file)
    lhh_jpg_folder_path = os.path.join(jpg_folder, lhh_name)

    convert_pdf_to_jpg(lhh_pdf_path, lhh_jpg_folder_path)