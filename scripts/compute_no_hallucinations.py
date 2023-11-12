import numpy as np
import sys, json
import argparse

def main(args):
    
    pairs = args.pairs.split(',')
    main_output = []

    for pair in pairs:

        with open(args.file_ref+'/'+pair+'/out.json','r') as f:
            results = json.load(f)

        output = []

        for key, value in results.items():
            if isinstance(value,dict) and value['chrf'] < args.thr:
                output.append(results[key])


        main_output.append(len(output))

        with open(args.file_ref+'/'+pair+'/hallucination.json','w') as f:
            json.dump(output,f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_ref", type=str, default="",
                        help="The path to the prediction directory")
    parser.add_argument("--thr", type=float, default=20,
                        help="The threshold for hallucination")
    parser.add_argument("--pairs", type=str, default="",
                        help="language pairs")
    args = parser.parse_args()
    main(args)
