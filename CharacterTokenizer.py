class CharacterTokenizer:
    def __init__(self, training_text: str) -> None:
        self.strtoint = {}
        self.inttostr = {}

        characters = sorted(set(training_text))
        
        for i, character in enumerate(characters):
            self.strtoint[character] = i
            self.inttostr[i] = character

    def vocab_size(self) -> int: 
        return len(self.strtoint)

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of token IDs."""
        token_ids = []

        for c in text: 
            token_ids.append(self.strtoint[c])

        return token_ids
        
    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back into text."""
        characters = []

        for i in token_ids:
            characters.append(self.inttostr[i])
        
        return "".join(characters)
