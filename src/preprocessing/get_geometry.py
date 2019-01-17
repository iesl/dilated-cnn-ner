import json
import os
import unicodedata
import pprint


rootdir = '../../../data/corpus-bioarxiv-extracted-texts'
targetdir='../../../data/pg_geometry.json'
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
            print len(pg_dict)
print len(pg_dict)
print pg_dict

with open(targetdir,'w') as write_file:
    json.dump(pg_dict,write_file,indent=4)
            

