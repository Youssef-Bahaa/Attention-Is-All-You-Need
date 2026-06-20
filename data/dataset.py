from datasets import load_dataset

ds = load_dataset("bentrevett/multi30k")
print(ds["train"][0])