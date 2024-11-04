import argparse
from transformers import AutoTokenizer
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor

from transformers import TextStreamer
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html
import string

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

# set default cuda device as 1
torch.cuda.set_device(1)

 
translation_table = str.maketrans(punctuation_dict)


def get_image_files(image_folder):
    image_files = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                image_files.append(os.path.join(root, file))
    return image_files

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()

    

    model.to(device='cuda',  dtype=torch.bfloat16)


    # TODO vary old codes, NEED del 
    image_processor = BlipImageEvalProcessor(image_size=1024)

    image_processor_high =  BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256


    image_files = get_image_files(args.image_folder)
    images = [load_image(image_file) for image_file in image_files]


    w, h = images[0].size
    # print(image.size)
    
    if args.type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '

    if args.box:
        bbox = eval(args.box)
        if len(bbox) == 2:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
            bbox[2] = int(bbox[2]/w*1000)
            bbox[3] = int(bbox[3]/h*1000)
        if args.type == 'format':
            qs = str(bbox) + ' ' + 'OCR with format: '
        else:
            qs = str(bbox) + ' ' + 'OCR: '

    if args.color:
        if args.type == 'format':
            qs = '[' + args.color + ']' + ' ' + 'OCR with format: '
        else:
            qs = '[' + args.color + ']' + ' ' + 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs



    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(prompt)


    inputs = tokenizer([prompt])


    # copy all images
    images_1 = [image.copy() for image in images]
    images = [image_processor(image) for image in images]
    # image_tensor = torch.stack(image_tensor).cuda()


    images_1 = [image_processor_high(image) for image in images_1]
    # image_tensor_1 = torch.stack(image_tensor_1).cuda()


    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    with torch.autocast("cuda", dtype=torch.bfloat16):
        for image, image_1 in zip(images, images_1):
            images = [(image.unsqueeze(0).half().cuda(), image_1.unsqueeze(0).half().cuda())]
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=False,
                num_beams = 1,
                no_repeat_ngram_size = 20,
                streamer=streamer,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--box", type=str, default= '')
    parser.add_argument("--color", type=str, default= '')
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()

    eval_model(args)
