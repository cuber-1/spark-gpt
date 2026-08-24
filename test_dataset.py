from CharacterTokenizer import CharacterTokenizer
from text_dataset import TextDataset


text = "hello world\n"
tokenizer = CharacterTokenizer(text)
dataset = TextDataset(text, tokenizer, context_length=4)

x, y = dataset[0]

print(x)
print(y)
print(tokenizer.decode(x.tolist()))
print(tokenizer.decode(y.tolist()))
