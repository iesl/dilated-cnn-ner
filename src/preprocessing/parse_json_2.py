from __future__ import print_function
import json
import os
import unicodedata
import argparse

arg_parser = argparse.ArgumentParser(description='Preprocess arxiv header data.')
arg_parser.add_argument('--input_dir', type=str, help='Data to process')
arg_parser.add_argument('--geometry_file', type=str, help='Page geometry file')
arg_parser.add_argument('--output_file', type=str, help='File to write')
args = arg_parser.parse_args()


escape_chars = ["\n", "\r", "\t", "\\", "\'", "\"", "\a", "\b", "\f", "\v"]


def check_escape_chr(ch):
    if ch in escape_chars or ch.isspace():
        return False
    else:
        return True

rootdir = args.input_dir
pg_file = args.geometry_file
target_file = args.output_file

target_string=""
unique_label={}
file_count=0
total_doc_count=0
with open(pg_file,'r') as pgf:
    pg_geometry=json.load(pgf)
with open(target_file, 'w') as output_file:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file!=".DS_Store":
                file_count+=1
                path=os.path.join(subdir, file)
                print(path)

                with open(path,'r') as f:
                    ipf=json.load(f)

                final_string=""
                count=0
                for ip in ipf:
                    count+=1
                    # print "count:",count
                    annotation=ip['annotations']
                    doc_id=ip['stableId']
                    page_x,page_y=pg_geometry[doc_id][2],pg_geometry[doc_id][3]
                    print(page_x, page_y)
                    page_records={}
                    all_tokens=[]
                    all_pages=[]
                    for annot in annotation:
                        off_labels={}
                        body=annot['body']
                        if body and ("ExternalTeam" in annot['annotPath']):
                            labels=body['labels']
                            clabels=labels['cellLabels']
                            tmp_arr=[]
                            offsets=[]
                            if len(clabels)>0:
                                for c in clabels:
                                    c2=c[1]

                                    if len(c2)>0:
                                        for i in c2:
                                            tmp=i[0]
                                            tmp_arr+=["I-"+tmp[0]]*(tmp[2])
                                            offsets+=range(tmp[1],tmp[1]+tmp[2])
                                    else:
                                        tmp_arr+=["I-"+c[0][0]]*(c[0][2])
                                        offsets+=range(c[0][1],c[0][1]+c[0][2])
                                    off_labels={offsets[i]:tmp_arr[i] for i in range(len(tmp_arr))}

                                rows = body['rows']
                                for r in rows:
                                    offset_local = r['offset']
                                    token_local = ''

                                    loci_local = r['loci']
                                    curr_token = 0
                                    start_x,start_y,end_x,end_y=0,0,0,0
                                    for l in range(len(loci_local)):
                                        if "g" in loci_local[l]:
                                            if check_escape_chr(loci_local[l]['g'][0][0]):
                                                token_local+=loci_local[l]['g'][0][0]
                                                token_local=unicodedata.normalize("NFKD",token_local)
                                                if start_x==0 and start_y==0:
                                                    start_x,start_y=loci_local[l]['g'][0][2][0],loci_local[l]['g'][0][2][1]
                                                    local_page=loci_local[l]['g'][0][1]
                                        if ("i" in loci_local[l]):
                                            if "g" in loci_local[l-1] and check_escape_chr(loci_local[l-1]['g'][0][0]):
                                                end_x=loci_local[l-1]['g'][0][2][0]+loci_local[l-1]['g'][0][2][2]
                                                end_y=loci_local[l-1]['g'][0][2][1]+loci_local[l-1]['g'][0][2][3]


                                                if offset_local in off_labels:
                                                    label_local=off_labels[offset_local]
                                                else:
                                                    label_local="0"
                                                all_tokens.append((token_local,start_x,start_y,end_x,end_y,label_local,local_page))
                                                start_x,start_y=0,0
                                                token_local=''
                                        elif (l==(len(loci_local)-1)):
                                            end_x=loci_local[l]['g'][0][2][0]+loci_local[l]['g'][0][2][2]
                                            end_y=loci_local[l]['g'][0][2][1]+loci_local[l]['g'][0][2][3]

                                            if offset_local in off_labels:
                                                label_local=off_labels[offset_local]
                                            else:
                                                label_local="0"
                                            all_tokens.append((token_local,start_x,start_y,end_x,end_y,label_local,local_page))
                                            start_x,start_y=0,0
                                            token_local=''

                            else:
                                label=annot['label']
                                rows=body['rows']
                                for r in rows:
                                    offset_local=r['offset']
                                    token_local=''

                                    loci_local=r['loci']
                                    curr_token=0
                                    # for l in loci_local:
                                    #     if "g" in l:
                                    #
                                    #         token_local[curr_token]+=l['g'][0][0]
                                    #     else:
                                    #         curr_token+=1
                                    start_x,start_y,end_x,end_y=0,0,0,0
                                    for l in range(len(loci_local)):
                                        # if l==len(loci_local)-1:
                                        #     print "last"
                                        # if l==0:
                                        #     start_x,start_y=loci_local[l]['g'][0][2][0],loci_local[l]['g'][0][2][1]
                                        #     local_page=loci_local[l]['g'][0][1]
                                        if "g" in loci_local[l]:
                                            #print "character:",loci_local[l]['g'][0][0]

                                            if check_escape_chr(loci_local[l]['g'][0][0]):
                                                token_local+=loci_local[l]['g'][0][0]
                                                token_local=unicodedata.normalize("NFKD",token_local)
                                                if start_x==0 and start_y==0:
                                                    start_x,start_y=loci_local[l]['g'][0][2][0],loci_local[l]['g'][0][2][1]
                                                    local_page=loci_local[l]['g'][0][1]
                                            #print token_local
                                        if ("i" in loci_local[l]):
                                            # print token_local
                                            #print "character:",loci_local[l]['i'][0][0]
                                            # print "elif",token_local
                                            if "g" in loci_local[l-1]:
                                                end_x=loci_local[l-1]['g'][0][2][0]+loci_local[l-1]['g'][0][2][2]
                                                end_y=loci_local[l-1]['g'][0][2][1]+loci_local[l-1]['g'][0][2][3]

                                            # if offset_local in off_labels:
                                            #     label_local=off_labels[offset_local]
                                            # else:
                                            #     label_local="0"
                                                label_local="I-"+annot['label']
                                                all_tokens.append((token_local,start_x,start_y,end_x,end_y,label_local,local_page))
                                                start_x,start_y=0,0
                                                token_local=''
                                        elif (l==(len(loci_local)-1)):
                                            # print token_local
                                            #print "character:",loci_local[l]['i'][0][0]
                                            # print "elif",token_local

                                            end_x=loci_local[l]['g'][0][2][0]+loci_local[l]['g'][0][2][2]
                                            end_y=loci_local[l]['g'][0][2][1]+loci_local[l]['g'][0][2][3]

                                            # if offset_local in off_labels:
                                            #     label_local=off_labels[offset_local]
                                            # else:
                                            #     label_local="0"
                                            label_local="I-"+annot['label']
                                            all_tokens.append((token_local,start_x,start_y,end_x,end_y,label_local,local_page))
                                            start_x, start_y = 0, 0
                                            token_local = ''

                    page_wise = {}
                    for ele in all_tokens:
                        if ele[6] not in all_pages:
                            all_pages.append(ele[6])
                            page_wise[ele[6]] = []
                            page_wise[ele[6]].append(list(ele)[:-1])
                        else:
                            page_wise[ele[6]].append(list(ele)[:-1])

                    for pg in all_pages:
                        doc_line = "0:0:" + str(page_x) + ":" + str(page_y)
                        print(doc_line, file=output_file)
                        tokens = page_wise[pg]
                        for ele in tokens:
                            if ele[1] != 0 and ele[2] != 0:
                                line = "%s %d:%d:%d:%d * %s" % (ele[0], ele[1], ele[2], ele[3], ele[4], ele[5])
                                print(line, file=output_file)
                                # target_string+=(str(ele[0])+" "+str(ele[1])+":"+str(ele[2])+":"+str(ele[3])+":"+str(ele[4])+" * "+str(ele[5]))
                                # target_string += "\n"
                        print(file=output_file)
                        # target_string += "\n"

                    # print(target_string, file=output_file)
                    # target_string = ""


