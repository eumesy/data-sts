import subprocess

import joblib
import pandas as pd
import stanfordnlp
from tqdm import tqdm

subprocess.run(
    ["wget", "-nc", "https://storage.googleapis.com/paws/english/paws_qqp.tar.gz"]
)
subprocess.run(["tar", "xf", "paws_qqp.tar.gz"])
subprocess.run(
    ["wget", "-nc", "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"]
)
subprocess.run(
    [
        "python",
        "qqp_generate_data.py",
        "--original_qqp_input",
        "quora_duplicate_questions.tsv",
        "--paws_input",
        "paws_qqp/dev_and_test.tsv",
        "--paws_output",
        "paws_qqp/dev_and_test_restored.tsv",
    ]
)


nlp = stanfordnlp.Pipeline(processors="tokenize,mwt")


def tokenize(text):
    return [t.text for s in nlp(text).sentences for t in s.tokens]


df = pd.read_csv("paws_qqp/dev_and_test_restored.tsv", sep="\t").dropna()

# strip `b"`
df["sentence1"] = df.sentence1.str[2:-1]
df["sentence2"] = df.sentence2.str[2:-1]

df[["sentence1", "sentence2"]].to_csv(
    "STS.input.paws-qqp.txt", header=False, index=False, sep="\t"
)
df.label.to_csv("STS.gs.paws-qqp.txt", header=False, index=False, sep="\t")

s1_tokens = [tokenize(s) for s in tqdm(df.sentence1)]
s2_tokens = [tokenize(s) for s in tqdm(df.sentence2)]
joblib.dump(list(zip(s1_tokens, s2_tokens)), "STS.input.paws-qqp.words.en_ewt.joblib")
