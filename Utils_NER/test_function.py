from ner_utils import ner_utils

ner_utils.convert_doccano2conll(
    doccano_path = '/Users/trungnt108.tech/NTT/NER_utils/Data/admin.jsonl',
    conll_path = '/Users/trungnt108.tech/NTT/NER_utils/Data/Trung_NER_bio.txt',
    BIO = True
)