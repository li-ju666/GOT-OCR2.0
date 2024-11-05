import os
from pdf2image import convert_from_path
import argparse


def convert_pdf_to_jpg(inputpdf_path, outputjpg_folder):
    os.makedirs(outputjpg_folder, exist_ok=True)
    images = convert_from_path(inputpdf_path)
    for i, image in enumerate(images):
        image_path = os.path.join(outputjpg_folder, f'{i+1:05}.jpg')
        image.save(image_path, 'JPEG')
        # print(f'Saved: {image_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf', help='input pdf file')
    parser.add_argument('image_folder', help='output folder')
    args = parser.parse_args()
    convert_pdf_to_jpg(args.pdf, args.image_folder)
