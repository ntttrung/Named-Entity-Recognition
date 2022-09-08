from ner_utils import ner_utils

ner_utils.convert_doccano2conll(
    doccano_path = '/Users/trungnt108.tech/NTT/Named-Entity-Recognition/Utils_NER/Data/output.jsonl',
    conll_path = '/Users/trungnt108.tech/NTT/Named-Entity-Recognition/Utils_NER/Data/ALL_NER_bio.txt',
    BIO = True
)