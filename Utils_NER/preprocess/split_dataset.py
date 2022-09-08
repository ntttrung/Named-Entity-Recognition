from sklearn.model_selection import train_test_split


file = open('/Users/trungnt108.tech/NTT/Named-Entity-Recognition/Utils_NER/Data/ALL_NER_bio.txt', 'r')
data = file.read().split('\n\n')[:-1]
train, test = train_test_split(data, test_size=0.25, random_state=42)
print(len(train), len(test))

##Train file
file_train = open('/Users/trungnt108.tech/NTT/Named-Entity-Recognition/dataset/final-dataset/train_data.txt', 'w')
for i in train:
    file_train.write(i + '\n\n')
file_train.close()

##Test file
file_test = open('/Users/trungnt108.tech/NTT/Named-Entity-Recognition/dataset/final-dataset/test_data.txt', 'w')
for i in test:
    file_test.write(i + '\n\n')
file_test.close()

##Val file
file_val = open('/Users/trungnt108.tech/NTT/Named-Entity-Recognition/dataset/final-dataset/val_data.txt', 'w')
for i in test:
    file_val.write(i + '\n\n')
file_val.close()

file.close()
