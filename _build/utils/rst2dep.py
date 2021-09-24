#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Script to convert Rhetorical Structure Theory trees from .rs3 format
to a dependency representation.
"""


import re, io, ntpath
from xml.dom import minidom
from collections import defaultdict
from xml.parsers.expat import ExpatError
from argparse import ArgumentParser
try:
    from .feature_extraction import ParsedToken
except ImportError:
    from feature_extraction import ParsedToken

class SIGNAL:
    def __init__(self, sigtype, sigsubtype, tokens):
        self.type = sigtype
        self.subtype = sigsubtype
        self.tokens = tokens

    def __repr__(self):
        return self.type + "/" + self.subtype + " (" + self.tokens + ")"

    def __str__(self):
        return "|".join([self.type,self.subtype,self.tokens])

class NODE:
    def __init__(self, id, left, right, parent, depth, kind, text, relname, relkind):

        """Basic class to hold all nodes (EDU, span and multinuc) in structure.py and while importing"""

        self.id = id
        self.parent = parent
        self.left = left
        self.right = right
        self.depth = depth
        self.dist = 0
        self.domain = 0  # Minimum sorted covering multinuc domain priority
        self.kind = kind #edu, multinuc or span node
        self.text = text #text of an edu node; empty for spans/multinucs
        self.token_count = text.count(" ") + 1
        self.relname = relname
        self.relkind = relkind #rst (a.k.a. satellite), multinuc or span relation
        self.sortdepth = depth
        self.children = []
        self.leftmost_child = ""
        self.dep_parent = ""
        self.dep_rel = relname
        self.tokens = []
        self.parse = ""
        self.signals = []

    def rebuild_parse(self):
        token_lines = []
        for tok in self.tokens:
            # prevent snipped tokens from pointing outside EDU
            if int(tok.head) < int(self.tokens[0].id):
                tok.head = "0"
            elif int(tok.head) > int(self.tokens[-1].id):
                tok.head = "0"
            token_lines.append("|||".join([tok.id,tok.text,tok.lemma,tok.pos,tok.pos,tok.morph,tok.head,tok.func,"_","_"]))
        self.parse = "///".join(token_lines)

    def out_conll(self,feats=False):
        tokens = self.text.split()
        self.rebuild_parse()
        head_word = "_"
        if len(self.tokens) == 0:  # No token information
            self.tokens.append(ParsedToken("1","_","_","_","_","0","_"))
        head_func = "_"
        snippet = self.text

        if feats:
            first_pos = "pos1=" + self.tokens[0].pos
            for tok in self.tokens:
                if tok.head == "0" and not tok.func == "punct":
                    head_word = "head_tok="+tok.lemma.replace("=","&eq;")
                    head_func = "head_func="+tok.func
                    head_pos = "head_pos="+tok.pos
                if tok.pos in ["PRP", "PP"]:
                    pro = "pro"
                else:
                    pro = "nopro"
            if self.tokens[0].text.istitle():
                caps = "caps"
            else:
                caps = "nocaps"
            last_tok = self.tokens[-1].lemma
            if self.heading == "head":
                self.heading = "heading=heading"
            if self.caption== "caption":
                self.heading = "caption=caption"
            if self.para== "open_para":
                self.para = "para=para"
            if self.item== "open_item":
                self.item = "item=item"
            if self.list== "ordered":
                self.list = "list=ordered"
            if self.list== "unordered":
                self.list = "list=unordered"
            if self.caption== "date":
                self.heading = "date=date"
            if self.subord in ["LEFT","RIGHT"]:
                self.subord = "subord=" + self.subord
            feats = "|".join(feat for feat in [first_pos, head_word, head_pos, "stype="+self.s_type, "len="+str(len(self.tokens)), head_func, self.subord, self.heading, self.caption, self.para, self.item, self.date] if feat != "_")
            if len(feats)==0:
                feats = "_"
        else:
            feats = "_"

        signals = ";".join([str(sig) for sig in self.signals]) if len(self.signals) > 0 else "_"
        #return "\t".join([self.id, snippet, str(self.dist),str(self.depth), str(self.domain), feats, self.dep_parent, self.dep_rel, "_", signals])
        return "\t".join([self.id, snippet, str(self.dist),"_", "_", feats, self.dep_parent, self.dep_rel, "_", signals])

    def out_malt(self):
        first = self.tokens[0].lemma
        first_pos = self.tokens[0].pos
        self.rebuild_parse()
        head_word = "_"
        for tok in self.tokens:
            if tok.head == "0" and not tok.func == "punct":
                head_word = tok.lemma
                head_func = tok.func
                head_pos = tok.pos
            if tok.pos in ["PRP", "PP"]:
                pro = "pro"
            else:
                pro = "nopro"
        if self.tokens[0].text.istitle():
            caps = "caps"
        else:
            caps = "nocaps"
        last_tok = self.tokens[-1].lemma
        feats = "|".join(feat for feat in [str(len(self.tokens)), head_func, self.subord, self.heading, self.caption, self.para, self.item, self.date] if feat != "_")
        if len(feats)==0:
            feats = "_"

        return "\t".join([self.id, first, head_word, self.s_type, first_pos, feats, self.dep_parent, self.dep_rel, "_", self.parse])

    def __repr__(self):
        return "\t".join([str(self.id),str(self.parent),self.relname,self.text])


def get_left_right(node_id, nodes, min_left, max_right, rel_hash):
    """
    Calculate leftmost and rightmost EDU covered by a NODE object. For EDUs this is the number of the EDU
    itself. For spans and multinucs, the leftmost and rightmost child dominated by the NODE is found recursively.
    """
    if nodes[node_id].parent != "0" and node_id != "0":
        parent = nodes[nodes[node_id].parent]
        if min_left > nodes[node_id].left or min_left == 0:
            if nodes[node_id].left != 0:
                min_left = nodes[node_id].left
        if max_right < nodes[node_id].right or max_right == 0:
            max_right = nodes[node_id].right
        if nodes[node_id].relname == "span":
            if parent.left > min_left or parent.left == 0:
                parent.left = min_left
            if parent.right < max_right:
                parent.right = max_right
        elif nodes[node_id].relname in rel_hash:
            if parent.kind == "multinuc" and rel_hash[nodes[node_id].relname] =="multinuc":
                if parent.left > min_left or parent.left == 0:
                    parent.left = min_left
                if parent.right < max_right:
                    parent.right = max_right
        get_left_right(parent.id, nodes, min_left, max_right, rel_hash)


def get_depth(orig_node, probe_node, nodes):
    """
    Calculate graphical nesting depth of a node based on the node list graph.
    Note that RST parentage without span/multinuc does NOT increase depth.
    """
    if probe_node.parent != "0":
        parent = nodes[probe_node.parent]
        if parent.kind != "edu" and (probe_node.relname == "span" or parent.kind == "multinuc" and probe_node.relkind =="multinuc"):
            orig_node.depth += 1
            orig_node.sortdepth +=1
        elif parent.kind == "edu":
            orig_node.sortdepth += 1
        get_depth(orig_node, parent, nodes)


def get_distance(node, parent, nodes):
    if node.id == "34":
        a=4
    head = node.parent
    dist = 1
    encountered = {}
    while head != parent.id:
        encountered[head] = dist
        head = nodes[head].parent
        dist += 1
        if head == "0":
            break
    if head == "0":
        # common ancestor
        dist2 = 1
        head = parent.parent
        while head != "0":
            if head in encountered:
                if nodes[head].kind == "multinuc" and node.dep_rel.endswith("_m"):  # multinucs should have priority against tying incoming RST rels
                    dist2 -= 1
                return dist2 #+ encountered[head]
            else:
                dist2 += 1
                head = nodes[head].parent
        return dist2 #+ encountered[head]
    else:
        # direct ancestry
        return 0 # dist

def read_rst(data, rel_hash, as_text=False):
    if not as_text:
        data = io.open(data, encoding="utf8").read()
    try:
        xmldoc = minidom.parseString(data)
    except ExpatError:
        message = "Invalid .rs3 file"
        return message

    nodes = []
    ordered_id = {}
    schemas = []
    default_rst = ""

    # Get relation names and their types, append type suffix to disambiguate
    # relation names that can be both RST and multinuc
    item_list = xmldoc.getElementsByTagName("rel")
    for rel in item_list:
        relname = re.sub(r"[:;,]", "", rel.attributes["name"].value)
        if rel.hasAttribute("type"):
            rel_hash[relname + "_" + rel.attributes["type"].value[0:1]] = rel.attributes["type"].value
            if rel.attributes["type"].value == "rst" and default_rst == "":
                default_rst = relname + "_" + rel.attributes["type"].value[0:1]
        else:  # This is a schema relation
            schemas.append(relname)

    item_list = xmldoc.getElementsByTagName("segment")
    if len(item_list) < 1:
        return '<div class="warn">No segment elements found in .rs3 file</div>'

    id_counter = 0

    # Get hash to reorder EDUs and spans according to the order of appearance in .rs3 file
    for segment in item_list:
        id_counter += 1
        ordered_id[segment.attributes["id"].value] = id_counter
    item_list = xmldoc.getElementsByTagName("group")
    for group in item_list:
        id_counter += 1
        ordered_id[group.attributes["id"].value] = id_counter
    ordered_id["0"] = 0

    element_types = {}
    node_elements = xmldoc.getElementsByTagName("segment")
    for element in node_elements:
        element_types[element.attributes["id"].value] = "edu"
    node_elements = xmldoc.getElementsByTagName("group")
    for element in node_elements:
        element_types[element.attributes["id"].value] = element.attributes["type"].value

    id_counter = 0
    item_list = xmldoc.getElementsByTagName("segment")
    for segment in item_list:
        id_counter += 1
        if segment.hasAttribute("parent"):
            parent = segment.attributes["parent"].value
        else:
            parent = "0"
        if segment.hasAttribute("relname"):
            relname = segment.attributes["relname"].value
        else:
            relname = default_rst

        # Tolerate schemas, but no real support yet:
        if relname in schemas:
            relname = "span"

            relname = re.sub(r"[:;,]", "", relname)  # remove characters used for undo logging, not allowed in rel names
        # Note that in RSTTool, a multinuc child with a multinuc compatible relation is always interpreted as multinuc
        if parent in element_types:
            if element_types[parent] == "multinuc" and relname + "_m" in rel_hash:
                relname = relname + "_m"
            elif relname != "span":
                relname = relname + "_r"
        else:
            if not relname.endswith("_r") and len(relname) > 0:
                relname = relname + "_r"
        edu_id = segment.attributes["id"].value
        contents = segment.childNodes[0].data.strip()
        nodes.append(
            [str(ordered_id[edu_id]), id_counter, id_counter, str(ordered_id[parent]), 0, "edu", contents, relname])

    item_list = xmldoc.getElementsByTagName("group")
    for group in item_list:
        if group.attributes.length == 4:
            parent = group.attributes["parent"].value
        else:
            parent = "0"
        if group.attributes.length == 4:
            relname = group.attributes["relname"].value
            # Tolerate schemas by treating as spans
            if relname in schemas:
                relname = "span"

            relname = re.sub(r"[:;,]", "", relname)  # remove characters used for undo logging, not allowed in rel names
            # Note that in RSTTool, a multinuc child with a multinuc compatible relation is always interpreted as multinuc
            if parent in element_types:
                if element_types[parent] == "multinuc" and relname + "_m" in rel_hash:
                    relname = relname + "_m"
                elif relname != "span":
                    relname = relname + "_r"
            else:
                relname = ""
        else:
            relname = ""
        group_id = group.attributes["id"].value
        group_type = group.attributes["type"].value
        contents = ""
        nodes.append([str(ordered_id[group_id]), 0, 0, str(ordered_id[parent]), 0, group_type, contents, relname])

    elements = {}
    for row in nodes:
        elements[row[0]] = NODE(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], "")

    for element in elements:
        if elements[element].kind == "edu":
            get_left_right(element, elements, 0, 0, rel_hash)

    for element in elements:
        node = elements[element]
        get_depth(node,node,elements)

    for nid in elements:
        node = elements[nid]
        if node.parent != "0":
            elements[node.parent].children.append(nid)
            if node.left == elements[node.parent].left:
                elements[node.parent].leftmost_child = nid

    # Ensure left most multinuc children are recognized even if there is an rst dependent further to the left
    for nid in elements:
        node = elements[nid]
        if node.kind == "multinuc" and node.leftmost_child == "":
            min_left = node.right
            leftmost = ""
            for child_id in node.children:
                child = elements[child_id]
                if child.relname.endswith("_m"):  # Using _m suffix to recognize multinuc relations
                    if child.left < min_left:
                        min_left = child.left
                        leftmost = child_id
            node.leftmost_child = leftmost

    signal_list = xmldoc.getElementsByTagName("signal")
    for sig in signal_list:
        nid = str(sig.attributes["source"].value)
        if nid not in elements:
            raise IOError("A signal element refers to source " + nid + " which is not found in the document\n")
        elements[nid].signals.append(SIGNAL(sig.attributes["type"].value,sig.attributes["subtype"].value,sig.attributes["tokens"].value))

    return elements


def seek_other_edu_child(nodes, source, exclude, block):
    """
    Recursive function to find some child of a node which is an EDU and does not have the excluded ID

    :param nodes: dictionary of IDs to NODE objects
    :param source: the source node from which to traverse
    :param exclude: node ID to exclude as target child
    :param block: list of IDs for which children should not be traversed (multinuc right children)
    :return: the found child ID or None if none match
    """

    if source == "0":
        return None
    else:
        # Check if this is already an EDU
        if nodes[source].kind == "edu" and source != exclude and source not in block:
            return source
        # Loop through children of this node
        children_to_search = [child for child in nodes[source].children if child not in nodes[exclude].children and child not in block]
        if len(children_to_search)>0:
            if int(exclude) < int(children_to_search[0]):
                children_to_search.sort(key=lambda x: int(x))
            else:
                children_to_search.sort(key=lambda x: int(x), reverse=True)
        for child_id in children_to_search:
            # Found an EDU child which is not the original caller
            if nodes[child_id].kind == "edu" and child_id != exclude and (nodes[source].kind != "span" or nodes[child_id].relname == "span") and \
                    not (nodes[source].kind == "multinuc" and nodes[source].leftmost_child == exclude) and \
                    (nodes[nodes[child_id].parent].kind not in ["span","multinuc"]):
                    #not (nodes[child_id].parent == nodes[exclude].parent and nodes[source].kind == "multinuc" and int(child_id) > int(exclude)):  # preclude right pointing rel between multinuc siblings
                return child_id
            # Found a non-terminal child
            elif child_id != exclude:
                # If it's a span, check below it, following only span relation paths
                if nodes[source].kind == "span":
                    if nodes[child_id].relname == "span":
                        candidate = seek_other_edu_child(nodes, child_id, exclude, block)
                        if candidate is not None:
                            return candidate
                # If it's a multinuc, only consider the left most child as representing it topographically
                elif nodes[source].kind == "multinuc" and child_id == nodes[source].leftmost_child:
                    candidate = seek_other_edu_child(nodes, child_id, exclude, block)
                    if candidate is not None:
                        return candidate
    return None


def find_dep_head(nodes, source, exclude, block):
    parent = nodes[source].parent
    if parent != "0":
        if nodes[parent].kind == "multinuc":
            for child in nodes[parent].children:
                # Check whether exclude and child are under the same multinuc and exclude is further to the left
                if nodes[child].left > int(exclude) and nodes[child].left >= nodes[parent].left and int(exclude) >= nodes[parent].left:
                    block.append(child)
    else:
        # Prevent EDU children of root from being dep head - only multinuc children possible at this point
        for child in nodes[source].children:
            if nodes[child].kind == "edu":
                block.append(child)
    candidate = seek_other_edu_child(nodes, nodes[source].parent, exclude, block)
    if candidate is not None:
        return candidate
    else:
        if parent == "0":
            return None
        else:
            if parent not in nodes:
                raise IOError("Node with id " + source + " has parent id " + parent + " which is not listed\n")
            return find_dep_head(nodes, parent, exclude, block)


def get_nonspan_rel(nodes,node):
    if node.parent == "0":  # Reached the root
        return "ROOT"
    elif nodes[node.parent].kind == "multinuc" and nodes[node.parent].leftmost_child == node.id:
        return get_nonspan_rel(nodes, nodes[node.parent])
    elif nodes[node.parent].kind == "multinuc" and nodes[node.parent].leftmost_child != node.id:
        return node#.relname
    elif nodes[node.parent].relname != "span":
        grandparent = nodes[node.parent].parent
        if grandparent == "0":
            return "ROOT"
        elif not (nodes[grandparent].kind == "multinuc" and nodes[node.parent].left == nodes[grandparent].left):
            return nodes[node.parent]#.relname
        else:
            return get_nonspan_rel(nodes,nodes[node.parent])
    else:
        if node.relname.endswith("_r"):
            return node#.relname
        else:
            return get_nonspan_rel(nodes,nodes[node.parent])


def make_rsd(rstfile, xml_dep_root,as_text=False, docname=None):
    nodes = read_rst(rstfile,{},as_text=as_text)
    out_graph = []
    if rstfile.endswith("rs3"):
        out_file = rstfile.replace(".rs3",".rsd")
    else:
        out_file = rstfile + ".rsd"
    if docname is not None:
        out_file = docname + ".rsd"

    dep_root=xml_dep_root
    if dep_root != "":
        try:
            from .feature_extraction import get_tok_info
        except ImportError:
            from feature_extraction import get_tok_info
        conll_tokens = get_tok_info(ntpath.basename(out_file).replace(".rsd",""),xml_dep_root)
        feats = True
    else:
        feats = False

    # Add tokens to terminal nodes
    if isinstance(nodes,str):
        pass
    edus = list(nodes[nid] for nid in nodes if nodes[nid].kind=="edu")
    edus.sort(key=lambda x: int(x.id))
    token_reached = 0
    for edu in edus:
        if dep_root != "":
            edu.tokens = conll_tokens[token_reached:token_reached+edu.token_count]
            edu.s_type = edu.tokens[0].s_type
            edu.para = edu.tokens[0].para
            edu.item = edu.tokens[0].item
            edu.caption = edu.tokens[0].caption
            edu.heading = edu.tokens[0].heading
            edu.list = edu.tokens[0].list
            if any(tok.date != "_" for tok in edu.tokens):
                edu.date = "date"
            else:
                edu.date = "_"
            start_tok = int(edu.tokens[0].id)
            end_tok = int(edu.tokens[-1].id)
            subord = "_"

            for tok in edu.tokens:
                if int(tok.head) < start_tok and tok.head != "0":
                    subord = "LEFT"
                    tok.head = "0"
                elif int(tok.head) > end_tok and tok.head != "0":
                    subord = "RIGHT"
                    tok.head = "0"

            edu.subord = subord

            # TODO: remove hard-wired GUM genres
            if "news" in out_file:
                edu.genre = "news"
            elif "whow" in out_file:
                edu.genre = "whow"
            elif "voyage" in out_file:
                edu.genre = "voyage"
            elif "interview" in out_file:
                edu.genre = "interview"
            elif "reddit" in out_file:
                edu.genre = "reddit"
            elif "academic" in out_file:
                edu.genre = "academic"
            elif "_bio" in out_file:
                edu.genre = "bio"
            elif "_fiction" in out_file:
                edu.genre = "fiction"
            else:
                edu.genre = "news"

            token_reached += edu.token_count

    # Get each node with 'span' relation its nearest non-span relname
    for nid in nodes:
        node = nodes[nid]
        if nid == "9":
            pass
        new_rel = node.relname
        sigs = []
        if node.parent == "0":
            new_rel = "ROOT"
        elif node.relname == "span" or (nodes[node.parent].kind == "multinuc" and nodes[node.parent].leftmost_child == nid):
            new_rel = get_nonspan_rel(nodes,node)
            if new_rel != "ROOT":
                sigs = new_rel.signals
                new_rel = new_rel.relname
        node.dep_rel = new_rel
        if len(sigs) > 0:
            node.signals = sigs

    for nid in nodes:
        node = nodes[nid]
        if nid == "9":
            pass
        if node.kind == "edu":
            dep_parent = find_dep_head(nodes,nid,nid,[])
            if dep_parent is None:
                #This is the root
                node.dep_parent = "0"
                node.dep_rel = "ROOT"
            else:
                node.dep_parent = dep_parent
            out_graph.append(node)

    # Get all multinuc ancestors of each node to compute multinuc domains they belong to
    multinuc_ancestors = defaultdict(list)
    domains = set()
    for nid in nodes:
        node = nodes[nid]
        if node.kind == "multinuc":
            for i in range(node.left,node.right+1):
                domain = (node.right-node.left,node.left,node.right)
                multinuc_ancestors[str(i)].append(domain)
                domains.add(domain)
    domains = sorted(list(domains),reverse=True)
    domain_mapping = {}
    max_domain = 1
    for domain in domains:
        domain_mapping[domain] = max_domain
        max_domain += 1

    # Get path distance from child to parent including phrase structure
    for nid in nodes:
        if nid == "3":
            s=3
        node = nodes[nid]
        node.domain = 0
        if node.dep_rel == "ROOT":
            node.dist = "0"
            continue
        if node.kind == "edu":
            parent = nodes[node.dep_parent]
            node.dist = get_distance(node, parent, nodes)
            relevant_domains = [d for d in multinuc_ancestors[node.id] if int(parent.id) >= d[1] and int(parent.id) <= d[2]]
            if len(relevant_domains) > 0:
                min_domain = sorted(relevant_domains)[0]
                node.domain = domain_mapping[min_domain]

    out_graph.sort(key=lambda x: int(x.id))

    out_mode = "conll"  # NB only in split mode below

    output = []

    for node in out_graph:
        if out_mode == "conll":
            output.append(node.out_conll(feats=feats))
        else:
            output.append(node.out_malt())

    return "\n".join(output) + "\n"


if __name__ == "__main__":
    desc = "Script to convert Rhetorical Structure Theory trees \n from .rs3 format to a dependency representation.\nExample usage:\n\n" + \
            "python rst2dep.py <INFILES>"
    parser = ArgumentParser(description=desc)
    parser.add_argument("infiles",action="store",help="file name or glob pattern, e.g. *.rs3")
    parser.add_argument("-r","--root",action="store",dest="root",default="",help="optional: path to corpus root folder containing a directory dep/ and \n"+
                                                           "a directory xml/ containing additional corpus formats")

    options = parser.parse_args()

    inpath = options.infiles

    if "*" in inpath:
        from glob import glob
        files = glob(inpath)
    else:
        files = [inpath]

    for rstfile in files:
        output = make_rsd(rstfile,options.root)
        with io.open(rstfile.replace("rs3","rsd"),'w',encoding="utf8",newline="\n") as f:
            f.write(output)

