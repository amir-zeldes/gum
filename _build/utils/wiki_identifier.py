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
        if fields[5] == '_':
            continue
        titles = fields[5].split('|')
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

not_found = 0
time = date.today()
with io.open('wiki_map.tab', 'w', encoding='utf8', newline="\n") as f:
    f.write('#FormattedTitle\tURLTitle\tWikidataID\tDateUpdated\n')
    for k, t in seen.items():
        if not t:
            # print(k)
            not_found += 1
        if k in match:
            f.write(f'{k}\t{match[k]}\t{t}\t{time}\n')
        else:
            f.write(f'{k}\t{k}\t{t}\t{time}\n')
# print(f'{not_found} titles are not found in wiki db.')
