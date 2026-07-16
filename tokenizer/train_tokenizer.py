from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>"
    )
)

tokenizer.pre_tokenizer = ByteLevel()


trainer = BpeTrainer(
    vocab_size=65536,
    special_tokens=[
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>"
    ]
)


tokenizer.train(
    [
        "data/processed/corpus.txt"
    ],
    trainer
)


tokenizer.save(
    "tokenizer/tokenizer.json"
)


print(
    "vocab:",
    tokenizer.get_vocab_size()
)
