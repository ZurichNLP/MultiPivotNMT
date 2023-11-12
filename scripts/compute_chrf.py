import sacrebleu
import numpy as np
import sys, json
import os
import argparse

def main(args):
    pairs = args.pairs.split(",")
    for pair in pairs:
        refs = []
        with open(args.base_path_input + "/" + pair + "/" + args.file_ref, 'r') as f:
            for line in f:
                refs.append(line.strip())

        preds = []
        with open(args.base_path_input + "/" + pair + "/" + args.file_pred, 'r') as f:
            for line in f:
                preds.append(line.strip())

        result = {}
        chrf = 0
        for i, (ref, pred) in enumerate(zip(refs,preds)):
            chrf_pred = (sacrebleu.sentence_chrf(pred, [ref], 6, 3)).score
            chrf += chrf_pred
            result[i] = {'pred':pred,'ref':ref,'chrf':chrf_pred}

        result['chrf_total'] = chrf * 1.0 / len(refs)

        print(f'Average ChrF is {result["chrf_total"]}')
        if not os.path.exists(args.base_path_output + "/" + pair):
            os.makedirs(args.base_path_output + "/" + pair)
        
        with open(args.base_path_output + "/" + pair + "/" + "out.json",'w') as f:
            json.dump(result,f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_ref", type=str, default="",
                        help="The name of the reference file")
    parser.add_argument("--file_pred", type=str, default=None,
                        help="The name of the prediction file")
    parser.add_argument("--pairs", type=str, default="",
                        help="language pairs")
    parser.add_argument('--base_path_input', type=str, default="",
                           help='Path to the prediction directory')
    parser.add_argument('--base_path_output', type=str, default="",
                           help='Do simple Ensembling')
    args = parser.parse_args()
    main(args)
