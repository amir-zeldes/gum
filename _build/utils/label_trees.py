from nltk.tree import Tree
from depedit import DepEdit
import io, os, sys, re
from glob import glob

depedit_precprocess = """{$tmpadv}=/now|earlier|ago|still|recently|already|currently|never|often|later|again|soon|always|meanwhile|once|previously|yet|far|ever|usually|first|long|longer|immediately|eventually|sometimes|annually|meanwhile|initially|then|early|frequently|late|finally|before|temporarily|ahead|formerly|prior|ultimately|lately|rarely|shortly|anymore|daily|previously|occasionally|recently|already|sooner|twice|regularly|thereafter|suddenly|repeatedly|simultaneously|eventually|constantly|subsequently|forever|since|anytime|afterward|originally|promptly|sometime|someday|seldom|indefinitely|monthly|continually|overnight|semiannually|routinely|presently|instantly|seasonally|ultimately|afterwards|nowadays|prematurely/
{$mnradv}=/well|quickly|sharply|directly|heavily|closely|publicly|easily|strongly|jointly|widely|rapidly|aggressively|fully|tentatively|hard|better|steadily|badly|effectively|formally|substantially|significantly|quietly|clearly|illegally|together|properly|slowly|carefully|fast|successfully|seriously|separately|privately|gradually|slightly|swiftly|actively|faster|officially|greatly|consistently|automatically|openly|smoothly|broadly|unanimously|freely|voluntarily|poorly|simply|indirectly|abruptly|firmly|harder|efficiently|financially|deliberately|nervously|desperately|readily|severely|differently|drastically|profitably|incorrectly|completely|improperly|suddenly|frantically|barely|evenly|specifically|fiercely|modestly|unfairly|best|dramatically|cheaply|normally|narrowly|unsuccessfully|adequately|painfully|fairly|gently|safely|cautiously|honestly|vehemently|vigorously|legally|tight|accurately|considerably|bitterly|positively|entirely|electronically|favorably|comfortably|neatly|promptly|partly|literally|adversely|further|prominently|reluctantly|Separately|especially|fundamentally|outright|alone|definitively|right|negatively|perfectly|jealously|reasonably|arbitrarily|pointedly|strictly|responsibly|similarly|artificially|accordingly|staunchly|somehow|otherwise|permanently|partially|emphatically|tacitly|busily|precisely|coolly|deeply|exactly|mildly|editorially|wildly|wisely|satisfactorily|altogether|materially|proudly|independently|enthusiastically|easy|internationally|spontaneously|firsthand|lightly|overwhelmingly|relentlessly|vividly|strenuously|mistakenly|personally|happily|hardest|harshly|summarily|explicitly|increasingly/
{$extverb}=/rise|raise|reduce|increase|decrease|decline|exceed|grow|drop|plummet|differ|slip|expand|soar|plunge/
{$datverb}=/sell|give|send|pay|award|offer|owe|provide|lend|make|issue|lease|bring|allocate|pass|present|deal|grant|write|deliver|transfer|extend|tender|supply|guarantee|call|cede|deny/
{$bnfverb}=/provide|make|buy|build|create|design|arrange|gather|leave|print|manufacture|do|fix|earn|open|pick|win|store|order|draft/
{$gumcleftsubj}=/Burns|Barbara|noggins|girls|coyotes|Senate|Olmec/
text=/{$gumcleftsubj}/;func=/cop/;func=/nsubj|expl/&lemma=/it/;func=/a(dv)?cl:relcl/\t#1>#2;#1>#3;#1>#4\t#3:func=cleft
xpos=/V.G/;func=/nsubj/\t#1>#2\t#1:func2=hassubj
xpos=/V.G/&func2=/(.*)/;func=/mark/\t#1>#2\t#1:func2=hasmark$1
xpos=/V.G/&func=/csubj/;func2!=/.*hassubj.*/\t#1>#2\t#1:func=nomsbj
xpos=/V.G/&func2!=/.*hassubj.*/&func!=/nomsbj/;func=/mark/\t#1>#2\t#1:func=nom
xpos=/V.G/&func=/advcl/&func2!=/.*hasmark.*/&func2!=/.*hassubj.*/\tnone\t#1:func=vbgadv
func=/advcl|nom/;func=/mark/&lemma=/when|while/\t#1>#2\t#2:func=stmp
text=/.*/;func=/cop/\t#1>#2\t#1:func=prd
text=/.*/;func=/i?obj/;func=/xcomp/\t#1>#2;#1>#3\t#3>#2;#2:func=nsubj;#3:func=prd
xpos=/JJ.*/&func=/xcomp/\tnone\t#1:func=prd
func=/advcl/;xpos=/TO/\t#1>#2\t#2:func=mark_prp
func=/advcl/;func=/mark/\t#1>#2\t#2:func=mark_adv
func=/dislocated/;xpos=/.*/\t#1.*#2\t#1:func=tpc
lemma=/{$mnradv}/&xpos=/RB.*/\tnone\t#1:func=mnradv
lemma=/{$tmpadv}/&xpos=/RB.*/\tnone\t#1:func=tmpadv
lemma=/put/;func=/obl/;func=/case/&lemma=/in|on|into|at|out|above|under|down|over|behind|through|toward|upon/\t#1>#2>#3\t#3:func=put
morph=/.*time.*/&func=/nmod|obl/;func=/case/\t#1>#2\t#2:func=pptmp
morph=/.*place.*/&func=/nmod|obl/;func=/case/&lemma=/from|to|into|towards?|across|onto|along|through/\t#1>#2\t#2:func=ppdir
func=/nmod|obl/;func=/case/&lemma=/into|towards?|onto/\t#1>#2\t#2:func=ppdir
morph=/.*place.*/&func=/nmod|obl/;func=/case/&lemma!=/of/\t#1>#2\t#2:func=pploc
morph=/.*time.*/&func=/prd/;func=/case/\t#1>#2\t#2:func=pptmp
morph=/.*place.*/&func=/prd/;func=/case/&lemma!=/of/\t#1>#2\t#2:func=pploc
lemma=/{$datverb}/&xpos=/V.*/;func=/obl/;lemma=/to/&func=/case/\t#1>#2>#3\t#3:func=ppdat
lemma=/{$bnfverb}/&xpos=/V.*/;func=/obl/;lemma=/for/&func=/case/\t#1>#2>#3\t#3:func=ppbnf
lemma=/do/;lemma=/so/&func=/advmod/\t#1>#2\t#2:func=prd
func=/(nmod|obl):unmarked/&lemma=/^([0-9]|[0-9][0-9]|[0-9][0-9][0-9]|(1|2)[0-9][0-9][0-9]|[12]?[0-9]:[0-5][0-9]|30th|a\.m\.|AD|afternoon|age|April|August|autumn|b\.i\.d\.|century|couple|day|decade|EDT|evening|fall|February|five|Friday|GMT|hour|hundred|January|July|June|length|life|March|match|matter|May|minute|moment|Monday|month|morning|next|night|number|oclock|October|p\.m\.|period|pm|q\.d\.|q\.o\.d\.|quarter|rest|Saturday|season|second|semester|September|spring|Su|summer|Summer|Sunday|that|thing|Thursday|tide|time|today|Today|tomorrow|tonight|Tuesday|Wednesday|week|weekday|weekend|while|winter|year|yesterday)$/\tnone\t#1:func=$1:tmod
func=/(obl|nmod):(npmod|unmarked)/;func=/nummod/\t#1>#2\t#1:func=ext
lemma=/{$extverb}/&xpos=/V.*/;func=/obl/;lemma=/by/&func=/case/\t#1>#2>#3\t#3:func=ppext
"""

d = DepEdit()
d.quiet = True
d.read_config_file(depedit_precprocess.strip().split("\n"))

mapping = {("NP","nsubj"):"NP-SBJ", ("NP","nsubj:pass"):"NP-SBJ", ("NP","obl:agent"):"NP-LGS",
           ("WHNP", "nsubj"): "NP-SBJ", ("WHNP", "nsubj:pass"): "NP-SBJ",
            ("NX", "nsubj"): "NX-SBJ", ("NX", "nsubj:pass"): "NX-SBJ",
           #("NP","tpc"):"NP-TPC",
           ("NP","cleft"):"NP-CLF",
           ("NP","obl:npmod"):"NP-ADV",
           ("NP", "nmod:npmod"): "NP-ADV",
           ("NP","obl:unmarked"):"NP-ADV",
           ("NP", "nmod:unmarked"): "NP-ADV",
           ("NP", "ext"): "NP-EXT",
           ("PP", "ppext"): "PP-EXT",
           ("PP", "ppdat"): "PP-DTV",
           ("PP", "ppbnf"): "PP-BNF",
           ("NP", "obl:tmod"): "NP-TMP",
           ("NP", "nmod:tmod"): "NP-TMP",
           ("NP", "vocative"): "NP-VOC",
           ("VP","csubj"):"VP-SBJ",
           ("SBAR","csubj"):"SBAR-SBJ",
           ("VP","mark_prp"):"VP-PRP",
           ("SBAR","mark_adv"):"SBAR-ADV",
           ("NP","prd"):"NP-PRD",
           ("NX","prd"):"NX-PRD",
           ("ADJP","prd"):"ADJP-PRD",
           ("ADVP","prd"):"ADVP-PRD",
           ("VP","prd"):"VP-PRD",
           ("VP","vbgadv"):"VP-ADV",
           ("ADVP","tmpadv"):"ADVP-TMP",
           ("ADVP","mnradv"):"ADVP-MNR",
           ("VP","nomsbj"):"VP-NOM-SBJ",
           ("VP","nom"):"VP-NOM",
           ("PP","pptmp"):"PP-TMP",
           ("PP","pploc"):"PP-LOC",
           ("PP","put"):"PP-PUT",
           ("PP","ppdir"):"PP-DIR",
           ("WHADVP","stmp"):"WHADVP-TMP"
           }

percolating1 = {"VP-PRP":"S",
               "VP-NOM-SBJ":"S",
               "VP-NOM":"S",
               "VP-SBJ":"S",
               "VP-PRD":"S",
               "VP-ADV":"VP",
               "NP-CLF":"NP",
               "NP-SBJ":"NP",
               "NP-LGS":"NP",
               "NX-SBJ":"NP",
               "NX-PRD":"NP",
               "NP-PRD":"NP",
               "WHADVP-TMP":"SBAR",
               }

percolating2 = {"S-PRP":"SBAR",
                "S-PRD": "S",
                "NP-CLF": "S",
                "VP-ADV": "S",
                "NP-SBJ": "NP",
                "NP-LGS": "NP",
                "NP-PRD": "NP",
                "WHADVP-TMP": "S",
                }

percolating3 = {"S-PRD": "SBAR",
                "NP-PRD": "PP-LOC",
                }
percolating4 = {
                "NP-PRD": "PP",
                }


percolating = (percolating1,percolating2,percolating3,percolating4)

cleanup = {
    "VP-NOM":"VP",
    "VP-NOM-SBJ":"VP",
    "VP-PRD": "VP",
    "VP-ADV":"VP",
    "NP-CLF":"NP",
    #"WHADVP-TMP": "WHADVP",
           }


def percolate(node,parent):
    cat, lab = node.label().split("-",maxsplit=1)
    parent.set_label(parent.label() + "-" + lab)
    node.set_label(cat)


def get_parent_child(t, position):
    node = t[position]
    node_text = str(t[position][0])
    parent = t[tuple(list(position)[:-1])]
    return node, node_text, parent


def add_ptb_labels(ptb_string, entidep_string):

    ptb_string = re.sub(r"(\))\s*\n(\(ROOT)",r"\1\n\n\2",ptb_string)  # prevent missing newline between sentences
    trees = ptb_string.strip().split("\n\n")
    depedited = d.run_depedit(entidep_string)

    dep_sents = depedited.strip().split("\n\n")
    dep_funcs = []

    for sent in dep_sents:
        dep_funcs.append([line.split("\t")[7] for line in sent.split("\n") if "\t" in line and "." not in line.split("\t")[0]])

    out_sents = []
    for i, tree_str in enumerate(trees):
        funcs = dep_funcs[i]
        t = Tree.fromstring(tree_str)

        current_token = 0
        # Labeling
        for position in t.treepositions():
            node, node_text, parent = get_parent_child(t, position)
            if " " not in node_text and isinstance(node,Tree):  # Token
                func = funcs[current_token]
                if (parent.label(), func) in mapping:
                    parent.set_label(mapping[(parent.label(), func)])
                current_token += 1

        # Percolation
        for perc in percolating:
            for position in t.treepositions():
                node, node_text, parent = get_parent_child(t, position)
                if isinstance(node,Tree):  # non-terminal
                    if node.label() in perc:
                        if parent.label() == perc[node.label()]:
                            percolate(node,parent)

        # Cleanup
        for position in t.treepositions():
            node = t[position]
            if isinstance(node,Tree):
                if node.label() in cleanup:
                    node.set_label(cleanup[node.label()])

        out_sents.append(str(t))

    return "\n\n".join(out_sents)


if __name__ == "__main__":
    doc = "GUM_academic_discrimination"
    trees = io.open(os.sep.join(["..","target","const"]) + os.sep + doc + ".ptb",encoding="utf8").read()
    conllu = io.open(os.sep.join(["pepper","tmp","entidep"]) + os.sep + doc+".conllu",encoding="utf8").read()

    ptb_labeled = add_ptb_labels(trees, conllu)

    print(ptb_labeled)
