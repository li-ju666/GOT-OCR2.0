IMG_FOLDER=images
TXT_FOLDER=texts
PDF_FOLDER=pdfs

# create the text and image folder
mkdir -p $TXT_FOLDER
mkdir -p $IMG_FOLDER

# iterate over all the pdf files
for pdf_file in $PDF_FOLDER/*.pdf
do
    # # remove all the spaces in the pdf file name
    # # there might be multiple spaces in the file name
    # if [[ $pdf_file == *" "* ]]; then
    #     new_pdf_file=$(echo $pdf_file | tr -s ' ')
    #     mv "$pdf_file" $new_pdf_file
    #     pdf_file=$new_pdf_file
    # fi

    # image folder name
    image_folder=$IMG_FOLDER/$(basename $pdf_file .pdf)
    txt_folder=$TXT_FOLDER/$(basename $pdf_file .pdf)

    echo "Processing $pdf_file ..."

    # convert pdf to images
    python3 pdf2images.py $pdf_file $image_folder

    # run the OCR
    python3 demo/multiocr.py --model-name  GOT_weights --image-folder $image_folder --type ocr > $txt_folder.txt

    # remove the image folder
    rm -rf $image_folder

done
echo "Done!"
