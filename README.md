# data-sts

# Format

## sentence pairs

- `STS.input.foo.txt` (original)
    - TSV file including sentence pairs
    - each line is of the form: sent1[TAB]sent2

- ` STS.input.foo.words.en_ewt.joblib` (tokenized by StanformdNLP)
    - joblib formatted file
    - list of words pairs

        ```python
        [
            (['This', 'is', 'first', 'sentence', '.'], ['This', 'is', 'second', 'sentence', '.']),
            (['foo', 'bar', ...], ['hoge', 'fuga', ...]),
            ...
        ]
        ```

## gold scores

- `STS.gs.foo.txt`
    - each line gold scores (sentence similarities)



# List of Dataset

## STS-B

`data/stsbenchmark`

### Web

<http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark>

### Stats

<http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark>

### Log of Data Preparation (for reproducibility)

```bash
wget http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz -P data
tar xvzf data/Stsbenchmark.tar.gz -C data

cd data/stsbenchmark
cut -f5 sts-test.csv > STS.gs.sts-test.txt
cut -f 6-7 sts-test.csv > STS.input.sts-test.txt
cut -f5 sts-dev.csv > STS.gs.sts-dev.txt
cut -f 6-7 sts-dev.csv > STS.input.sts-dev.txt
cut -f5 sts-train.csv > STS.gs.sts-train.txt
cut -f 6-7 sts-train.csv > STS.input.sts-train.txt
```
