import json
from collections import defaultdict

data = json.load(open("align_gpt4o_zeroshot.json"))

docs_by_rank = defaultdict(list)

for doc in data:
    lists = []
    for source in data[doc]:
        lists.append(data[doc][source])

    # Fine the maximum number of lists inside 'lists' which are identical
    max_lists = 0

    for i in range(len(lists)):
        identical = 0
        for j in range(i+1, len(lists)):
            if sorted(lists[i]) == sorted(lists[j]):
                identical += 1
        if identical > max_lists:
            max_lists = identical

    docs_by_rank[max_lists].append(doc)


for rank in docs_by_rank:
    print(f"Rank: {rank+1} - {len(docs_by_rank[rank])}")

for rank in sorted(docs_by_rank.keys(), reverse=True):
    print(f"Rank: {rank}")
    for doc in docs_by_rank[rank]:
        print("\t" + doc)
    print()
