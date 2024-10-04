from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from argparse import ArgumentParser
import json
import copy
import torch
import os, sys


script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
connpreds_dir = script_dir + ".." + os.sep + "data" + os.sep + "connector_preds" + os.sep

#device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_test(data_path, checkpoint_path, out_path):
    test_data=[]
    print(f"Loading dataset from {data_path} ...")
    with open(data_path, 'r') as inp:
        for line in inp:
            line = line.strip()
            line = json.loads(line)
            if line not in test_data:
                test_data.append(line)
    print(f"Test dataset size: {len(test_data)}")

    model_id = checkpoint_path
    print(f"model path: {model_id}")

    # Load tokenizer of FLAN-t5-large
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # huggingface hub model id
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto')

    print("Prediction...")

    #import pdb;pdb.set_trace();
    my_copy = copy.deepcopy(test_data)
    for ii, samp in enumerate(tqdm(test_data)):
        inputs = tokenizer("generate connective: " + samp['input'], return_tensors="pt", padding="max_length",
                               truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, return_dict_in_generate=True)
        concs = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
        my_copy[ii]['connectors'] = concs

    with open(out_path, "w") as outf:
        outf.write('\n'.join(json.dumps(i) for i in my_copy))


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-i", "--input", help="input data file", default='./missing.jsonl')
    p.add_argument("-o", "--output", help="output file", default=connpreds_dir + "gum_implicit_add_preds.jsonl")
    p.add_argument("-m", "--model", help="model path", default="model/")
    p.add_argument("-d","--delete", help="delete contents of missing.jsonl after predicting", action="store_true")

    args = p.parse_args()

    if not os.path.exists("model/model.safetensors"):
        response = input("! connpred model missing - do you wish to download it?\n")
        if response.lower() in ["yes", "y"]:
            import requests
            import shutil

            # Download and unzip archive contents to model/
            url = "https://gucorpling.org/amir/download/gdtb/connpred_model.zip"
            r = requests.get(url, stream=True)
            with open("model/connpred_model.zip", "wb") as f:
                shutil.copyfileobj(r.raw, f)
            import zipfile
            with zipfile.ZipFile("model/connpred_model.zip", "r") as zip_ref:
                zip_ref.extractall("model/")
            sys.stderr.write("Downloaded connpred model files to model/\n")
        else:
            sys.exit(1)

    run_test(args.input, args.model, args.output)

    if args.delete:
        with open(args.input, "w") as f:
            f.write("")
