import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import json

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)



def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    result = []
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                ff = open("mesh_caps_ids.txt", "a+", encoding="utf8")
                ff.write(gen_i.strip())
                ff.write("\n")
                ff.close()
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
    json.dump(result, f, ensure_ascii=False)
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    # Write result in json format
    import json
    temp = []
    with open("/content/mesh_caps_ids.txt", "r") as f:
        for line in f:
            temp.append(line.replace("\n", ""))

    f = open("mesh_caps_ids_gts.json", "w+")
    results = []
    index = 0
    while index != len(temp):
        res = {"id": temp[index], "pred": temp[index + 1], "gts": temp[index + 2:index + 7]}
        results.append(res)
        index += 7
    json.dump(results, f, ensure_ascii=False)
    f.close()


    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    args = parser.parse_args()

    # Hardcode paths
    args.features_path = "/content/drive/MyDrive/ColabNotebooks/UIT-MeshedMemoryTransformer/uitviic_detections.hdf5"
    args.annotation_paths = "/content/drive/MyDrive/Images"
    args.m = 40

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_paths, args.annotation_paths)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab_.pkl', 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load('/content/drive/MyDrive/ColabNotebooks/UIT-MeshedMemoryTransformer/Model/original_best_real.pth')
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores)
