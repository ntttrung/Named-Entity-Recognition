import json

with open('/Users/trungnt108.tech/NTT/Named-Entity-Recognition/Utils_NER/Data/admin.jsonl 3', 'r') as json_file:
    json_list = list(json_file)


with open('/Users/trungnt108.tech/NTT/Named-Entity-Recognition/Utils_NER/Data/output.jsonl', 'w') as outfile:
    for json_str in json_list:
        result = json.loads(json_str)
        if len(result['label']) > 0:
            json.dump(result, outfile)
            outfile.write('\n')
