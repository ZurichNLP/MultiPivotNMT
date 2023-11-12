from mt_task import MTTask
from translation_models import load_translation_model
import argparse
import torch

convert_lang_code_id = ["af", "am", "ar", "hy", "as", "ast", "az", "be", "bn", "bs", "bg", "my", "ca", "ceb", "zh", "hr", "cs", "da", "nl", "en", "et", "tl", "fi", "fr", "ff", "gl", "lg", "ka", "de", "el", "gu", "ha", "he", "hi", "hu", "is", "ig", "id", "ga", "it", "ja", "jv", "kea", "kam", "kn", "kk", "km", "ko", "ky", "lo", "lv", "ln", "lt", "luo", "lb", "mk", "ms", "ml", "mt", "mi", "mr", "mn", "ne", "nso", "no", "ny", "oc", "or", "om", "ps", "fa", "pl", "pt", "pa", "ro", "ru", "sr", "sn", "sd", "sk", "sl", "so", "ku", "es", "sw", "sv", "tg", "ta", "te", "th", "tr", "uk", "umb", "ur", "uz", "vi", "cy", "wo", "xh", "yo", "zu"]

def get_pivots(language_pairs, result_path):

    results = {}
    with open(result_path, 'r') as f:
        for line in f:
            line = line.strip()
            results["-".join(line.split("-")[:2])] = float(line.split("-")[-1])

    pivots = []
    for lang_pair in language_pairs:
        curr_pivot = [convert_lang_code_id.index("en")]
        source_pivot = []
        source_index = convert_lang_code_id.index(lang_pair[0])
        for i in range(0,101):
            if i != source_index:
                source_pivot.append((str(source_index)+"-"+str(i), results[str(source_index)+"-"+str(i)]))
        source_pivot = sorted(source_pivot, key=lambda x: x[1],reverse=True)

        if int(source_pivot[0][0].split("-")[1]) == curr_pivot[0]:
            curr_pivot.append(int(source_pivot[1][0].split("-")[1]))
        else:
            curr_pivot.append(int(source_pivot[0][0].split("-")[1]))

        target_pivot = []
        target_index = convert_lang_code_id.index(lang_pair[1])
        for i in range(0,101):
            if i != target_index:
                target_pivot.append((str(i)+"-"+str(target_index), results[str(i)+"-"+str(target_index)]))
        target_pivot = sorted(target_pivot, key=lambda x: x[1],reverse=True)

        if int(target_pivot[0][0].split("-")[0]) not in curr_pivot:
            curr_pivot.append(int(target_pivot[0][0].split("-")[0]))
        elif int(target_pivot[1][0].split("-")[0]) not in curr_pivot:
            curr_pivot.append(int(target_pivot[1][0].split("-")[0]))
        else:
            curr_pivot.append(int(target_pivot[2][0].split("-")[0]))

        curr_pivot = [convert_lang_code_id[x] for x in curr_pivot]

        pivots.append(curr_pivot)

    return pivots

def get_pivots_bridge(language_pairs, result_path):

    results = {}
    with open(result_path, 'r') as f:
        for line in f:
            line = line.strip()
            results["-".join(line.split("-")[:2])] = float(line.split("-")[-1])

    pivots = []
    for lang_pair in language_pairs:
        curr_pivot = []
        source_pivot = []
        source_index = convert_lang_code_id.index(lang_pair[0])
        for i in range(0,101):
            if i != source_index:
                source_pivot.append((str(source_index)+"-"+str(i), results[str(source_index)+"-"+str(i)]))
        source_pivot = sorted(source_pivot, key=lambda x: x[1],reverse=True)
        curr_pivot.append(int(source_pivot[0][0].split("-")[1]))

        target_pivot = []
        target_index = convert_lang_code_id.index(lang_pair[1])
        for i in range(0,101):
            if i != target_index:
                target_pivot.append((str(i)+"-"+str(target_index), results[str(i)+"-"+str(target_index)]))
        target_pivot = sorted(target_pivot, key=lambda x: x[1],reverse=True)
        curr_pivot.append(int(target_pivot[0][0].split("-")[0]))

        curr_pivot = [convert_lang_code_id[x] for x in curr_pivot]

        pivots.append(curr_pivot)

    return pivots


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_translation_model(args.model_path, device=device)
    language_pairs = args.language_pairs.split(',')
    language_pairs = [x.split("-") for x in language_pairs]

    if args.pivots is None:

        if args.do_bridge:
            pivots = get_pivots_bridge(language_pairs,args.result_path)
        else:
            pivots = get_pivots(language_pairs,args.result_path)
    elif args.pivots == "top":
        pivots = [["en", "es", "fr"]] * len(language_pairs)
    elif args.pivots == "direct":
        pivots = [[None]]*len(language_pairs)
    elif args.pivots == "en":
        pivots = [["en"]] * len(language_pairs)
    else:
        pivots = args.pivots.split(',')
        pivots = [x.split("-") for x in pivots]

    tasks = []
    for lang_pair, pivot in zip(language_pairs, pivots):
        tasks.append(MTTask(lang_pair[0],lang_pair[1],pivot,args.result_path,args.input_path))
        print(f"Task added {lang_pair[0]} - {lang_pair[1]} - {pivot}")


    for task in tasks:
        if args.do_bridge:
            print(f"Bridge Translation {task}, pivot: {task.pivots}")
            if task.pivots[0] == task.pivots[1]:
                out_path = task.evaluate(model.translate_pivot,'bridge')
            else:
                out_path = task.evaluate(model.translate_bridge, 'bridge')
            print(f"Translations saved in {out_path}")
        else:
            if task.pivots[0] == None:
                print(f"Evaluating {task} direct")
                out_path = task.evaluate(model.translate, 'direct', False)
                print(f"Translations saved in {out_path}")
            elif len(task.pivots) == 1:
                print(f"Evaluating {task} pivot: {task.pivots}")
                out_path = task.evaluate(model.translate_pivot, 'pivot', False)
                print(f"Translations saved in {out_path}")
            elif args.do_simple_avg:
                print(f"Evaluating {task} multi-pivot: {task.pivots}")
                out_path = task.evaluate(model.translate_multi_pivot, 'pivot', True)
                print(f"Translations saved in {out_path} with simple averaging..")
            else:
                print(f"Evaluating {task} multi-pivot: {task.pivots}")
                out_path = task.evaluate(model.translate_multi_pivot, 'pivot', False)
                print(f"Translations saved in {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="",
                        help="The HF model path")
    parser.add_argument("--pivots", type=str, default=None,
                        help="pivots for each language pair")
    parser.add_argument("--language_pairs", type=str, default="",
                        help="language pairs")
    parser.add_argument('--do_bridge', default=False, action='store_true',
                           help='Do bridge between source and target')
    parser.add_argument('--do_simple_avg', default=False, action='store_true',
                           help='Do simple Ensembling')
    parser.add_argument("--result_path", type=str, default="m2m_100",
                        help="The output path")
    parser.add_argument("--input_path", type=str, default="gsarti/flores_101",
                        help="Input path to dataset")
    
    args = parser.parse_args()
    main(args)
