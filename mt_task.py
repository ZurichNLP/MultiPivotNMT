import logging
import subprocess
import tempfile
from pathlib import Path
from datasets import load_dataset
import os
from scripts.utils_run import FLORES101_CONVERT

class MTTask:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 pivots,
                 testset: str,
                 input_path: str,
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.pivots = pivots
        self.language_pair = f"{src_lang}-{tgt_lang}"
        self.testset = testset
        self.input_path = input_path
        base_out_dir = Path(__file__).parent / "out"
        if not base_out_dir.exists():
            base_out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = base_out_dir / self.testset
        self.out_dir.mkdir(exist_ok=True)

        self.out_dir = self.out_dir / self.language_pair
        self.out_dir.mkdir(exist_ok=True)
        self.load_converter = FLORES101_CONVERT

    def __str__(self):
        return f"{self.testset}-{self.src_lang}-{self.tgt_lang}"

    def evaluate(self, translation_method: callable, type='direct', simple_avg=False) -> Path:

        ## load FLORES dataset
        if self.input_path.endswith('.txt'):
            with open(self.input_path, 'r') as f:
                source_sentences = [line.rstrip() for line in f]
        else:
            source_sentences = load_dataset(self.input_path,self.load_converter[self.src_lang])['devtest']['sentence']

        if type == 'direct':
            translations = translation_method(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            source_sentences=source_sentences,
            )
        else:
            translations = translation_method(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            pivots=self.pivots,
            source_sentences=source_sentences,
            simple_avg=simple_avg,
            )

        if type == 'direct':
            file_name = 'direct'
        else:
            file_name = type + "-" +"-".join(self.pivots)

        with open(str(self.out_dir)+"/"+file_name+".txt", 'w') as f:
        #with tempfile.NamedTemporaryFile("w", dir=self.out_dir, delete=False) as f:
            #logging.info(f"Saving translations in {f.name}")
            f.write("\n".join(translations))

        if not os.path.isfile(str(self.out_dir)+"/"+"ref.text"):
            target_sentences = load_dataset('gsarti/flores_101', self.load_converter[self.tgt_lang])['devtest'][
                'sentence']
            with open(str(self.out_dir) + "/" + "ref.txt", 'w') as f:
                f.write("\n".join(target_sentences))


        #subprocess.call([
        #    "sacrebleu",
        #    "--test-set", self.testset,
        #    "--language-pair", self.language_pair,
        #    "--input", f.name,
        #    ])
        return Path(f.name)
