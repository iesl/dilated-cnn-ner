from __future__ import print_function
import json
import os
import unicodedata
import pprint
import argparse

arg_parser = argparse.ArgumentParser(description='Get page geometry from raw texts')
arg_parser.add_argument('--text_path', type=str, help='Path to extracted texts')
arg_parser.add_argument('--output_file', type=str, help='File to write')
args = arg_parser.parse_args()

#rootdir = '/iesl/canvas/strubell/data/arxiv-headers/corpus-bioarxiv-extracted-texts'
#rootdir = '/iesl/canvas/saunders/datasets/bioarxiv-project/corpus-bioarxiv-extracted-texts'
rootdir = args.text_path
targetdir=args.output_file
pg_dict={}
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file!=".DS_Store":
            path=os.path.join(subdir,file)
            #print path
            with open(path,'r') as f:
                ipf=json.load(f)
            sizes=ipf['pages'][0]['pageGeometry']
            doc_name=ipf['pages'][0]['textgrid']['stableId']
            pg_dict[doc_name]=sizes
            print(len(pg_dict))
print(len(pg_dict))
print(pg_dict)

with open(targetdir,'w') as write_file:
    json.dump(pg_dict,write_file,indent=4)
            

