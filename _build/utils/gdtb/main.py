import argparse
import os
from glob import glob
try:
    from .process import read_file
    from .convert import Converter
    from .utils import output_file
except:
    from process import read_file
    from convert import Converter
    from utils import output_file
from modules.cache import Cache
from modules.hypophora import Hypophora
from modules.explicit import Explicit
from modules.implicit import Implicit
from modules.altlex import Altlex
from modules.altlexC import AltlexC
from modules.entrel import EntRel
from modules.norel import NoRel
from argspan_ordering import order_rel_args, remove_duplicates


script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
gum_target = script_dir + ".." + os.sep + ".." + os.sep + "target" + os.sep
gum_target_conllu = script_dir + ".." + os.sep + ".." + os.sep + "target" + os.sep + "dep" + os.sep + "not-to-release" + os.sep
gum_target_rs4 = script_dir + ".." + os.sep + ".." + os.sep + "target" + os.sep + "rst" + os.sep + "rstweb" + os.sep
gum_target_gdtb = gum_target + "rst" + os.sep + "gdtb" + os.sep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input data dir", default='./data')
    parser.add_argument("-o", help="output dir", default=None)
    parser.add_argument("-d", help="document name or all if None", default=None)
    parser.add_argument("--cache", help="use cache or not (just auto predictions)", choices=["full", "filtered", "none"], default="full")
    parser.add_argument("--format", choices=["tab","rels","pdtb","all"], help="output format", default="all")
    args = parser.parse_args()

    data_dir = args.input
    output_dir = args.o
    if output_dir is None:
        output_dir = gum_target_gdtb

    # read json mapping
    disco_pred_dir = os.path.join(data_dir, 'discodisco_preds')
    explicit_mapping_dir = os.path.join(data_dir, 'mappings.json')
    # TODO: replace this with discodisco inference

    conn_pred_dir = os.path.join(data_dir, "connector_preds")

    # initialize conversion modules
    MODULES = {
        'cache': Cache(data_dir, explicit_mapping_dir, disco_pred_dir, filter=args.cache),
        'hypophora': Hypophora(data_dir, explicit_mapping_dir, disco_pred_dir),
        'explicit': Explicit(data_dir, explicit_mapping_dir, disco_pred_dir),
        'implicit': Implicit(data_dir, explicit_mapping_dir, disco_pred_dir, conn_pred_dir),
        'entrel' : EntRel(data_dir, explicit_mapping_dir, disco_pred_dir),
        'altlex': Altlex(data_dir, explicit_mapping_dir, disco_pred_dir),
        'altlexc': AltlexC(data_dir, explicit_mapping_dir)
    }
    norel = NoRel()

    # process each doc
    conllu_dir = gum_target_conllu
    rs4_dir = gum_target_rs4
    for doc_id, conllu_name in enumerate(glob(conllu_dir + "*.conllu")):
        if args.d is not None:
            if args.d not in conllu_name:
                continue
        docname = os.path.basename(conllu_name).replace(".conllu", "")
        print(docname)

        # process two files (conllu, rs4) and create a document state to store sentences, relations, edus, and spans
        doc_state = read_file(os.path.join(conllu_dir, conllu_name), os.path.join(rs4_dir, docname+".rs4"), docname)

        output = []
        doc_converter = Converter(doc_state, MODULES)

        # Give cache and altlexc modules the current doc state
        doc_converter.modules["cache"].set_doc_state(doc_state)
        doc_converter.modules["altlexc"].set_doc_state(doc_state)

        # Run conversion cascade for each input relation
        for node_id, rel in doc_state.rels.items():
            doc_converter.convert(rel, cache=args.cache)
            ordered = order_rel_args(rel,doc_state)
            if ordered is not None:
                if len(ordered) > 0:
                    output.extend(ordered)

        # Catch remaining NoRel cases
        output += norel.convert(doc_state)

        # Apply relations from cache module (potentially overwrites other assignments)
        for rel in MODULES["cache"].additional_rels[docname]:
            ordered = order_rel_args(rel, doc_state)
            if ordered is not None:
                if len(ordered) > 0:
                    output.extend(ordered)

        output = remove_duplicates(output)

        output.sort(key=lambda x: min(set(x[7]).union(x[8])))  # Order by lowest EDU ID

        # output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file(output_dir, output, doc_state, format=args.format)


if __name__ == "__main__":
    main()
