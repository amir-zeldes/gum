"""
dep2rst.py

Converts .rsd or .conllu style discourse dependencies to .rs3 discourse constituent trees. Exact constituent hierarchy can be
guessed using left or right attachment priority, or specified explicitly in column 3 of the input format, as produced
by the reverse script, rst2dep.py.

Input format - .rsd:
1	Greek court rules	0	_	_	_	2	attribution_r	_	_
2	worship of ancient Greek deities is legal	5	_	_	_	5	preparation_r	_	_
3	Monday , March 27 , 2006	4	_	_	_	5	circumstance_r	_	_
4	Greek court has ruled	0	_	_	_	5	attribution_r	_	_
5	that worshippers of the ancient Greek religion may now formally associate and worship at archeological sites .	0	_	_	_	0	ROOT	_	_
6	Prior to the ruling , the religion was banned from conducting public worship at archeological sites by the Greek Ministry of Culture .	1	_	_	_	5	background_r	_	_
7	Due to that , the religion was relatively secretive .	0	_	_	_	6	result_r	_	_
8	The Greek Orthodox Church , a Christian denomination , is extremely critical of worshippers of the ancient deities .	1	_	_	_	6	background_r	_	_
9	Today , about 100,000 Greeks worship the ancient gods , such as Zeus , Hera , Poseidon , Aphrodite , and Athena .	2	_	_	_	5	background_r	_	_
10	The Greek Orthodox Church estimates	0	_	_	_	11	attribution_r	_	_
11	that number is closer to 40,000 .	0	_	_	_	9	concession_r	_	_
12	Many neo - pagan religions , such as Wicca , use aspects of ancient Greek religions in their practice ;	3	_	_	_	5	background_r	_	_
13	Hellenic polytheism instead focuses exclusively on the ancient religions ,	0	_	_	_	12	contrast_m	_	_
14	as far as the fragmentary nature of the surviving source material allows .	0	_	_	_	13	concession_r	_	_

Input format - .conllu:
# text = Greek court rules worship of ancient Greek deities is legal
1	Greek	Greek	ADJ	JJ	Degree=Pos	2	amod	_	Discourse=attribution:1->2:0
2	court	court	NOUN	NN	Number=Sing	3	nsubj	_	_
3	rules	rule	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	_
4	worship	worship	NOUN	NN	Number=Sing	10	nsubj	_	Discourse=preparation:2->5:5
5	of	of	ADP	IN	_	8	case	_	_
6	ancient	ancient	ADJ	JJ	Degree=Pos	8	amod	_	_
7	Greek	Greek	ADJ	JJ	Degree=Pos	8	amod	_	_
8	deities	deity	NOUN	NNS	Number=Plur	4	nmod	_	_
9	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	10	cop	_	_
10	legal	legal	ADJ	JJ	Degree=Pos	3	ccomp	_	_

# text = Monday, March 27, 2006
1	Monday	Monday	PROPN	NNP	Number=Sing	0	root	_	Discourse=circumstance:3->5:4
2	,	,	PUNCT	,	_	4	punct	_	_
3	March	March	PROPN	NNP	Number=Sing	4	compound	_	_
...

"""

import io, sys
from argparse import ArgumentParser
from utils.rst2dep import NODE
from collections import defaultdict

# Default relations to include in generated .rs3 header
DEFAULT_RELATIONS = \
    {"rst":{"adversative-antithesis","attribution-positive","attribution-negative","context-background","causal-cause","context-circumstance",
            "adversative-concession","contingency-condition","elaboration-additional","elaboration-attribute","evaluation-comment",
            "explanation-evidence","explanation-justify","mode-manner","mode-means","explanation-motivation",
            "organization-phatic","organization-preparation","organization-heading","purpose-goal","purpose-attribute","topic-question",
            "restatement-partial","causal-result","topic-solutionhood"},
     "multinuc":{"joint-other","adversative-contrast","same-unit","joint-sequence","joint-disjunction","restatement-repetition","joint-list"}}


def conllu2rsd(conllu):
    lines = conllu.split("\n")
    edus = []
    words = []
    for line in lines:
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                continue
            if "Discourse=" in fields[-1]:
                if len(words) > 0:
                    edu = [edu_id, " ".join(words), dist, "_", "_", "_", parent, relname, "_", "_"]
                    edus.append("\t".join(edu))
                    words = []
                rel = [a for a in fields[-1].split("|") if a.startswith("Discourse")][0].split("=")[1]
                parts = rel.split(":")
                relname = parts[0]
                edge = parts[1]
                if len(parts) > 2:
                    dist = parts[2]
                else:
                    dist = "0"
                if not relname.endswith("_m"):
                    relname += "_r"
                if "->" in edge:
                    edu_id, parent = edge.split("->")
                else:
                    edu_id = edge
                    parent = "0"
            words.append(fields[1])
    if len(words) > 0:
        edu = [edu_id, " ".join(words), dist, "_", "_", "_", parent, relname, "_", "_"]
        edus.append("\t".join(edu))
    return "\n".join(edus) + "\n"


def xml_escape(edu_contents):
    return edu_contents.replace("&","&amp;").replace(">","&gt;").replace("<","&lt;")


def determinstic_groups(nodes):
    """
    Create an ID map with a deterministic ordering of group IDs based on a depth first climb of the ordered EDUs
    """
    edus = {e.id:e for e in nodes.values() if e.kind == "edu"}
    id_map = {int(e.id):int(e.id) for e in edus.values()}
    max_id = sorted([int(e.id) for e in edus.values()])[-1]
    for edu_id in sorted(edus,key=lambda x: int(edus[x].id)):
        parent = edus[edu_id].parent
        while parent != 0:
            if parent not in id_map:
                max_id += 1
                id_map[parent] = max_id
            parent = nodes[parent].parent
    return id_map


def rsd2rs3(rsd, ordering="dist", default_rels=False, strict=True):
    nodes = {}
    if default_rels:
        rels = DEFAULT_RELATIONS
    else:
        rels = {"rst":set(),"multinuc":set()}
    childmap = defaultdict(set)
    lines = rsd.split("\n")
    max_id = 0
    for line in lines:
        if "\t" in line:
            fields = line.split("\t")
            eid = fields[0]
            contents = fields[1]
            head = fields[6]
            dist = int(fields[2]) if fields[2] != "_" else 0
            depth = int(fields[3]) if fields[3] != "_" else 0
            domain = int(fields[4]) if fields[4] != "_" else 0
            reltype = "multinuc" if fields[7].endswith("_m") else "rst"
            relation = fields[7].replace("_m","").replace("_r","")
            if relation != "ROOT":
                if not default_rels:
                    rels[reltype].add(relation)
                elif default_rels and relation not in rels[reltype]:
                    sys.stderr.write("! Unlisted relation detected: " + relation + " (" + reltype + ")\n")
                    if strict:
                        sys.exit(0)
                    else:
                        relation = "span"
                        if reltype != "multinuc":
                            reltype = "span"
            node = NODE(int(eid),int(eid),int(eid),int(head),depth,"edu",contents,relation,reltype)
            node.dist = dist
            node.domain = domain
            node.dep_parent = int(head)
            nodes[int(eid)]= node
            if head != "0":
                childmap[int(head)].add(int(eid))
            max_id += 1

    # Compute path lengths to root
    for nid in nodes:
        p = nodes[nid].parent
        path_length = 0
        while p != 0:
            p = nodes[p].parent
            path_length += 1
        nodes[nid].depth = path_length

    # Add deterministic ordering prioritizing right or left children, if desired
    left_children = [n for n in sorted(nodes,key=lambda x: nodes[x].parent-nodes[x].left) if nodes[n].parent > nodes[n].id]
    right_children = [n for n in sorted(nodes,key=lambda x: nodes[x].left-nodes[x].parent) if nodes[n].parent < nodes[n].id]
    max_dist = 0
    if ordering != "dist":
        if ordering == "ltr":
            priority1 = left_children
            priority2 = right_children
        else:
            priority2 = left_children
            priority1 = right_children
        for nid in priority1 + priority2:
            nodes[nid].dist = max_dist
            max_dist += 1

    # Make span based tree, pretending multinucs are all rst relations
    nids_by_depth = sorted([n for n in nodes],reverse=True, key=lambda x:nodes[x].depth)
    level = 10000
    max_attached_dist = defaultdict(lambda : -1)
    top_span = {}
    span_by_dist = defaultdict(dict)
    for nid in nids_by_depth:
        if nodes[nid].depth < level:  # New level reached
            level = nodes[nid].depth
            level_nids_by_dist = sorted([n for n in nodes if nodes[n].depth==level and nodes[n].kind=="edu" and nodes[n].dep_parent!=0], key=lambda x:(nodes[x].dist,nodes[x].left))
            for nid2 in level_nids_by_dist:
                # Get top span for child and parent
                current_dist = nodes[nid2].dist
                if nid2 not in top_span:
                    top_span[nid2] = nodes[nid2]
                child = top_span[nid2]
                if nodes[nid2].dep_parent not in top_span:
                    top_span[nodes[nid2].dep_parent] = nodes[nodes[nid2].dep_parent]
                parent = top_span[nodes[nid2].dep_parent]

                if max_attached_dist[nodes[nid2].dep_parent] == nodes[nid2].dist and child.relkind =="multinuc":  # Multinuc sibling attachment
                    if current_dist == 0:
                        child.parent = nodes[nid2].dep_parent
                    else:
                        child.parent = span_by_dist[nodes[nid2].dep_parent][current_dist-1].id
                else:  # Attach and add span
                    max_id += 1
                    max_attached_dist[nodes[nid2].dep_parent] = nodes[nid2].dist
                    prev_rel = parent.relname
                    prev_kind = parent.relkind
                    parent.relname = "span"
                    parent.relkind = "span"
                    span = NODE(max_id, min(child.left, parent.left), max(child.right, parent.right), 0, parent.depth, "span", "", prev_rel, prev_kind)
                    child.parent = parent.id
                    parent.parent = max_id
                    nodes[max_id] = span
                    span_by_dist[nodes[nid2].dep_parent][current_dist] = span
                    top_span[nodes[nid2].dep_parent] = span

    # Convert multinuc spans to actual multinucs
    done = set()
    for nid in nodes:
        node = nodes[nid]
        if node.relkind == "multinuc" and node.id not in done:
            parent = nodes[node.parent]
            grandparent = nodes[parent.parent]
            node.parent = grandparent.id
            grandparent.kind = "multinuc"
            parent.relkind = "multinuc"
            parent.relname = node.relname
            done.add(parent.id)

    # Build header
    header = "<rst>\n\t<header>\n\t\t<relations>\n"
    rel_list = []
    for rel in rels["rst"]:
        rel_list.append('\t\t\t<rel name="' + rel +'" type="rst"/>')
    for rel in rels["multinuc"]:
        rel_list.append('\t\t\t<rel name="' + rel +'" type="multinuc"/>')
    header += "\n".join(sorted(rel_list)) + "\n\t\t</relations>\n\t</header>\n\t<body>\n"

    edus = [n for n in nodes.values() if n.kind == "edu"]
    groups = [n for n in nodes.values() if n.kind != "edu"]

    id_map = determinstic_groups(nodes)

    edus_out = []
    for edu in sorted(edus, key=lambda x:x.id):
        if edu.parent == 0:
            seg = '\t\t<segment id="' + str(edu.id) + '"/>'
        else:
            seg = '\t\t<segment id="'+str(edu.id)+'" parent="'+str(id_map[edu.parent])+'" relname="'+edu.relname+'">'+xml_escape(edu.text)+'</segment>'
        edus_out.append(seg)

    groups_out = []
    for group in sorted(groups, key=lambda x: id_map[x.id]):
        if group.parent == 0:
            seg = '\t\t<group id="' + str(id_map[group.id]) + '" type="' + group.kind + '"/>'
        else:
            seg = '\t\t<group id="'+str(id_map[group.id])+'" type="'+group.kind+'" parent="'+str(id_map[group.parent])+'" relname="'+group.relname+'"/>'
        groups_out.append(seg)

    output = header + "\n".join(edus_out) + "\n" + "\n".join(groups_out) + "\n\t</body>\n</rst>\n"

    return output


if __name__ == "__main__":
    desc = "Script to convert discourse dependencies to Rhetorical Structure Theory trees \n in the .rs3 format.\nExample usage:\n\n" + \
            "python dep2rst.py INFILE"
    p = ArgumentParser(description=desc)
    p.add_argument("file", help="discourse dependency file in .rsd or .conllu format")
    p.add_argument("-f","--format",choices=["rsd","conllu"],default="rsd",help="input format")
    p.add_argument("-d","--depth",choices=["ltr","rtl","dist"],default="dist",help="how to order depth")
    p.add_argument("-r","--rels",action="store_true",help="use DEFAULT_RELATIONS for the .rs3 header instead of rels in input data")

    opts = p.parse_args()
    data = io.open(opts.file,encoding="utf8").read()

    if opts.format == "conllu":
        data = conllu2rsd(data)

    output = rsd2rs3(data, ordering=opts.depth)

    print(output)
