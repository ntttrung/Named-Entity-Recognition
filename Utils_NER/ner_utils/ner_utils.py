import pickle

def read_pickle(path:str) -> list:
    '''
        Read file pkl to list
        ex:
            Input:
                path: 'demo.pkl'
            Output:
                [[('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O')],[...]]
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)

def write_pickle(data:list, path:str) -> None:
    '''
        Save list to pkl
        ex:
            Input:
                dt: [[('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O')],[...]]
                path: 'demo.pkl'
            Output:
                save list to flie 'demo.pkl'
    '''
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_conll(path:str) -> list:
    '''
        Read file conll to list
        ex:
            Input:
                path: 'conll.txt'
            Output:
                [[('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O')],[...]]
    '''
    out = []
    data = open(path,'r',encoding='utf-8').read().split('\n\n')
    for sq in data:
        temp = []
        for line in sq.split('\n'):
            if len(line) == 0:
                break
            w,t = line.split()
            temp.append((w,t))
        if len(temp) != 0:
            out.append(temp)
    return out

def write_conll(data:list, path:str) -> None:
    '''
        Save list to conll
        ex:
            Input:
                dt: [[('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O')],[...]]
                path: 'conll.txt'
            Output:
                save list to flie 'conll.txt'
    '''
    with open(path, 'w', encoding='utf-8') as f:
        for sq in data:
            for txt in sq:
                ww, tt = txt
                f.write(ww+'\t'+tt)
                f.write('\n')
            f.write('\n')
def convert_IOtag_to_BIOtag(data):
    '''
        Convert IOtag to BIOtag
        ex:
            Input:
                [[('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O'), (':', 'O'),('0212', 'PHONENUMBER'),('2808', 'PHONENUMBER'),('306', 'PHONENUMBER')],...]
            Output:
                [[('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O'), (':', 'O'),('0212', 'B-PHONENUMBER'),('2808', 'I-PHONENUMBER'),('306', 'I-PHONENUMBER')],...]
    '''
    out = []
    for line in data :
        newtag= []
        for i in range(len(line)):
            if line[i][1] != 'O':
                if i == 0 or line[i-1][1] != line[i][1]:
                    temp = 'B-' + line[i][1]
                elif i == len(line)-1 or  line[i+1][1] != line[i][1] :
                    temp = 'I-' + line[i][1]
                else:
                    temp = 'I-' + line[i][1]
            else:
                temp = line[i][1]
            newtag.append((line[i][0],temp))
        out.append(newtag)
    return out

def convert_BIOtag_to_IOtag(data):
    '''
        Convert BIOtag to IOtag
        ex:
            Input:
                [[('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O'), (':', 'O'),('0212', 'B-PHONENUMBER'),('2808', 'I-PHONENUMBER'),('306', 'I-PHONENUMBER')],...]
               
            Output:
                [[('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O'), (':', 'O'),('0212', 'PHONENUMBER'),('2808', 'PHONENUMBER'),('306', 'PHONENUMBER')],...]
    '''
    return [[(w,t.replace('B-','').replace('I-','')) for w,t in line] for line in data]

def convert_coll2span(line):
    out = []
    tag = None
    w_ = []
    for w,t in line:
        if t != 'O':
            if tag == None:
                tag = t
                w_.append(w)
            else:
                if tag != t:
                    out.append((' '.join(w_),tag))
                    w_ = [w]
                    tag = t
                else:
                    w_.append(w)
        else:
            if tag == None:
                out.append((w,t))
                tag = None
            else:
                if tag != t:
                    out.append((' '.join(w_),tag))
                    out.append((w,t))
                    w_ = []
                    tag = None
                # else:
                #     w_.append(w)
    if tag == None:
        pass
    else:
        out.append((' '.join(w_),tag))   
    return out  
 
def convert_IOtag_to_span(line:list) -> list:
    '''
        Convert IOtag to span
        ex:
            Input:
                [('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O'), (':', 'O'),('0212', 'PHONENUMBER'),('2808', 'PHONENUMBER'),('306', 'PHONENUMBER')]
            Output:
                [('Thông', 'O'),('tin', 'O'),('liên', 'O'),('hệ', 'O'), (':', 'O'),('0212 2808 306', 'PHONENUMBER')]
    '''
    out = []
    tag = None
    w_ = []
    for w,t in line:
        if t != 'O':
            if tag == None:
                tag = t
                w_.append(w)
            else:
                if tag != t:
                    out.append((' '.join(w_),tag))
                    w_ = [w]
                    tag = t
                else:
                    w_.append(w)
        else:
            if tag == None:
                out.append((w,t))
                tag = None
            else:
                if tag != t:
                    out.append((' '.join(w_),tag))
                    out.append((w,t))
                    w_ = []
                    tag = None
    if tag == None:
        pass
    else:
        out.append((' '.join(w_),tag))   
    return out  
def isBelongRange(a, b):
     return b[0] >= a[0] and b[1] <= a[1]
     
def sorted_idx(idx, isReverse=False):
    if len(idx) == 0:
        return idx
    temp = [(u, v, a) for u, v, a in sorted(idx, key=lambda item: item[0], reverse=isReverse)]
    res = [temp[0]]
    for i in range(1, len(temp)):
        if isBelongRange(temp[i], temp[i-1]):
            res[-1] = temp[i]
        elif not isBelongRange(temp[i-1], temp[i]):
            res.append(temp[i])
    return res
def convert_jsondict2list(line):
    text = line['text']
    label = sorted_idx(line['label'])
    res = []
    ress = []
    cur = 0
    for s, e, l in label:
        if cur < s:
            res.append([cur, s, 'O'])
        res.append([s, e, l])
        cur = e
    if cur < len(text):
        res.append([cur, len(text), 'O'])
    
    for s, e, l in res:
        for i in text[s:e].strip().split():
            ress.append((i, l))
    return ress
def get_label(data):
    count = 0
    out = []
    for w,t in data:
        if t!='O':
            start = count
            end = count + len(w)
            out.append([start,end,t])
        count+= len(w)+1
    return out
import json
def convert_doccano2conll(doccano_path: str, conll_path: str, BIO = True) -> None:
    '''
        Convert doccano to CONLL
        docano: https://github.com/doccano/doccano
        Input: JSONL
            {"data": "- Thay m\u1eb7t C\u00f4ng ty \u0111\u1ec3 l\u00e0m vi\u1ec7c v\u1edbi c\u00e1c c\u01a1 quan nh\u00e0 n\u01b0\u1edbc nh\u01b0 BHXH , Chi c\u1ee5c Thu\u1ebf , C\u01a1 quan Ki\u1ec3m to\u00e1n \u0111\u1ed9c l\u1eadp .", "label": [[60, 64, "ORGANIZATION"], [67, 79, "ORGANIZATION"], [82, 107, "ORGANIZATION"]], "ID": 6440, "SOURCE": "chotot_filtered_BIO"}
            {"data": "B\u1ea1n t\u00f4i c\u0169ng nh\u01b0 nhi\u1ec1u ng\u01b0\u1eddi , \u0111\u00e0nh l\u1ea5y ng\u00e0y c\u1ea3 nh\u00e0 \u0111i c\u00e1ch ly t\u1ea1i B\u1ec7nh vi\u1ec7n d\u00e3 chi\u1ebfn s\u1ed1 6 l\u00e0 l\u1ea7n cu\u1ed1i qu\u00e2y qu\u1ea7n c\u00f3 m\u1eb9 .", "label": [[67, 90, "LOCATION"]], "ID": 4033, "SOURCE": "vnexpress_374_BIO"}
            ...
        Output: CONLL
            IOtag or BIOtag
    '''
    IO_data = []
    with open(doccano_path, 'r', encoding='utf-8') as f:
        for i in list(f):
            temp = convert_jsondict2list((json.loads(i)))
            st = 0
            if len(temp) > 0:
                for i, (w, t) in enumerate(temp):
                    if w == '.' and t == 'O':
                        IO_data.append(temp[st:i+1])
                        st = i+1
                    if i == len(temp)-1 and w != '.':
                        temp.append(('.', 'O'))
    if BIO:
        BIO_data = convert_IOtag_to_BIOtag(IO_data)
        write_conll(BIO_data, conll_path)
    else:
        write_conll(IO_data, conll_path)
def convert_CONLL2doccano(conll_path, doccano_path, start_id=10000, source='uknow'):
    IOtag = read_conll(conll_path)
    count = start_id
    lst_json = []
    for line in IOtag:
        count += 1
        temp = dict()
        dt = convert_IOtag_to_span(line)
        label = get_label(dt)
        temp['ID'] = count
        temp["SOURCE"] = source
        temp["data"] = " ".join([w for w,_ in line])
        temp["label"] = label
        lst_json.append(temp)
    with open(doccano_path, 'w',encoding='utf-8') as outfile:
        for entry in lst_json:
            json.dump(entry, outfile)
            outfile.write('\n')