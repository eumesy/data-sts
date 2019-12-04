import subprocess

import joblib
import pandas as pd
import stanfordnlp
from tqdm import tqdm

subprocess.run(
    ["wget", "-nc", "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"]
)

nlp = stanfordnlp.Pipeline(processors="tokenize,mwt")


def tokenize(text):
    return [t.text for s in nlp(text).sentences for t in s.tokens]


df = pd.read_csv("quora_duplicate_questions.tsv", sep="\t").dropna()
df[["question1", "question2"]].to_csv(
    "STS.input.qqp.txt", header=False, index=False, sep="\t"
)
df.is_duplicate.to_csv("STS.gs.qqp.txt", header=False, index=False, sep="\t")

q1_tokens = [tokenize(s) for s in tqdm(df.question1)]
q2_tokens = [tokenize(s) for s in tqdm(df.question2)]
joblib.dump(list(zip(q1_tokens, q2_tokens)), "STS.input.qqp.words.en_ewt.joblib")
