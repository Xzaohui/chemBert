# from pathlib import Path

# from tokenizers import ByteLevelBPETokenizer

# paths = ['data/smiles.txt']

# # Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()

# # Customize training
# tokenizer.train(files=paths, vocab_size=10000, min_frequency=2, special_tokens=[
#     "<s>",
#     "<pad>",
#     "</s>",
#     "<unk>",
#     "<mask>",
# ])

# # Save files to disk
# tokenizer.save_model("./tokenizer", "")

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./tokenizer/chemical_tokenizer-vocab.json",
    "./tokenizer/chemical_tokenizer-merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("CC(=O)OC(CC(=O)O)C[N+](C)(C)C").ids)