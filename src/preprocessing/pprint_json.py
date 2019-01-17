import json
import os
import unicodedata
import pprint

escape_chars=["\n","\r","\t","\\","\'","\"","\a","\b","\f","\v"]


def check_escape_chr(ch):
    if ch in escape_chars or ch.isspace():
        return False
    else:
        return True

import os
rootdir = '../../../data/datasets/UmaCzi2018-annotation-exports'
# rootdir2='../../../data/corpus-bioarxiv-250-plus-gold'
target_file='../../input_file_3.txt'
target_string=""
unique_labels={}
file_count=0
total_doc_count=0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file!=".DS_Store":
            file_count+=1
            path=os.path.join(subdir, file)
            # path2=os.path.join(rootdir2,file[:-12],"textgrid.json")
            print path
            # print path2

            with open(path,'r') as f:
                ipf=json.load(f)
            # pp=pprint.pprint(ipf,indent=4)
            # a=pp.pprint(ipf)
            with open(path,'w') as f2:
                f2.write(json.dumps(ipf,indent=2))
