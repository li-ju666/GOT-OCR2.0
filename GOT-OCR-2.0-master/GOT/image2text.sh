IMG_FOLDER=images
TXT_FOLDER=texts

# create the text folder
mkdir -p $TXT_FOLDER

# iterate over all folders in the image folder
for image_folder in $IMG_FOLDER/*; do
    # text folder name
    txt_folder=$TXT_FOLDER/$(basename $image_folder)
    python3 demo/multiocr.py --model-name  GOT_weights --image-folder $image_folder --type ocr > $txt_folder.txt
done
