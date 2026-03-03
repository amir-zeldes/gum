#!/usr/bin/env python

"""
This script maps Wikipedia article headings used as Wikification identifiers in GUM
to Wikidata IDs, for example mapping Theseus -> Q1320718

Note that you must first download and compile the database required by wikimapper:

pip install wikimapper
mkdir data
wikimapper download enwiki-latest --dir data
wikimapper create enwiki-latest --dumpdir data --target data/index_enwiki-latest.db
python wiki_identifier.py

For more information see https://pypi.org/project/wikimapper/
"""

import io, os
from datetime import date
from wikimapper import WikiMapper
mapper = WikiMapper("data" + os.sep + "index_enwiki-latest.db")

tsv_dir = os.sep.join(['..','target','coref','tsv'])
seen = {}
match = {}
for filename in os.listdir(tsv_dir):
    if not filename.endswith('.tsv'): continue
    lines = io.open(tsv_dir+os.sep+filename, encoding='utf8').read().split('\n')
    for line in lines:
        if '\t' not in line: continue
        fields = line.split('\t')
        if fields[6] == '_':
            continue
        titles = fields[6].split('|')
        for title in titles:
            title = title.split('[')[0]
            search = title.replace("%2D","-").replace("%2C",",").replace("%29",")").replace("%28","(")
            if title not in seen:
                if search == 'David_Enoch_(Philosopher)':
                    search = 'David_Enoch_(philosopher)'
                    match[title] = 'David_Enoch_(philosopher)'
                if search == 'Frame_semantics(linguistics)':
                    search = 'Frame_semantics_(linguistics)'
                    match[title] = 'Frame_semantics_(linguistics)'
                if search == 'Tulsa_Riverside_Airport':
                    search = 'Richard_Lloyd_Jones_Jr._Airport'
                    match[title] = 'Richard_Lloyd_Jones_Jr._Airport'
                wikidata_id = mapper.title_to_id(search)
                if not wikidata_id:
                    new_search = search[0].upper() + search[1:]
                    wikidata_id = mapper.title_to_id(new_search)
                    if wikidata_id:
                        match[title] = new_search
                seen[title] = wikidata_id

known_missing = {"Toys_%22R%22_Us":"Q696334",
                 "MCI_Telecommunications_Corp._v._AT&T_Co.":"Q117459825",
                 "Property_Rights_%28economics%29":"Q8799101",
                 "Shanlyn_A._S._Park":"Q123588026",
                 "Gil_Penalosa":"Q61947420",
                 "The_i_Paper":"Q1943651",
                 "Exposition_du_système_du_monde":"Q19170514",
                 "Gemini_%28chatbot%29":"Q116698014",
                 "Sally_Meen":"Q7405228",
                 "Robert_Lyons_Danly":"Q100375920",
                 "William_Lansing_Gleason": "Q8014313"
                 }

not_found = 0
time = date.today()
with io.open('wiki_map.tab', 'w', encoding='utf8', newline="\n") as f:
    f.write('#FormattedTitle\tURLTitle\tWikidataID\tDateUpdated\n')
    for k, t in sorted(seen.items()):
        if not t:
            if k in known_missing:
                f.write(f'{k}\t{k}\t{known_missing[k]}\t{time}\n')
                continue
            else:
                print(k)
                not_found += 1
        if k in match:
            f.write(f'{k}\t{match[k]}\t{t}\t{time}\n')
        else:
            f.write(f'{k}\t{k}\t{t}\t{time}\n')
print(f'{not_found} titles are not found in wiki db.')
