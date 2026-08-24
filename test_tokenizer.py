
from CharacterTokenizer import CharacterTokenizer

training_text = "hello world\n"
tokenizer = CharacterTokenizer(training_text)

encoded = tokenizer.encode(training_text)
decoded = tokenizer.decode(encoded)

assert decoded == training_text
assert tokenizer.vocab_size() == len(set(training_text))

print(encoded)
print(decoded)

