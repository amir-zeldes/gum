from typing import List
import os, io


ellipsis_marker = "<*>"
type_map = {"explicit":"Explicit","implicit":"Implicit","entrel":"EntRel","altlex":"AltLex","altlexc":"AltLexC","norel":"NoRel","hypophora":"Hypophora"}


def format_range(tok_ids):
    # Takes a list of IDs and returns formatted string:
    # contiguous subranges of numbers are separated by '-', e.g. 5-24
    # discontinuous subranges are separated by ',', e.g. 2,5-24
    def format_subrange(subrange):
        if len(subrange) == 1:
            return str(subrange[0]+1)
        else:
            return str(min(subrange)+1) + "-" + str(max(subrange)+1)

    subranges = [[]]
    last = None
    for tid in sorted(tok_ids):
        if last is None:
            subranges[-1].append(tid)
        elif tid == last +1:
            subranges[-1].append(tid)
        else:
            subranges.append([tid])
        last = tid

    formatted = []
    for subrange in subranges:
        formatted.append(format_subrange(subrange))

    return ",".join(formatted)


def format_text(arg1_toks, toks):
    last = arg1_toks[0] - 1
    output = []
    for tid in sorted(arg1_toks):
        if tid != last + 1:
            output.append(ellipsis_marker)
        output.append(toks[tid].text)
        last = tid
    return " ".join(output)


def format_standoff(span_tokens, token_offsets):
    """
    Format a list of possibly discontinuous token IDs into a standoff character offset string.
    Each discontinuous span is represented as start offset, .. , end offset. Multiple spans are separated by a semicolon.
    Example: span_tokens: [1,2,3,5,6,7], token_offsets: {1:(0,2),2:(3,5),3:(6,8),5:(9,11),6:(12,14),7:(15,17)}
    Returns: "0..8;9..17"

    :param span_tokens: list of token IDs, one-based index
    :param token_offsets: dict mapping token IDs to character offsets, zero-based character indices
    :return: formatted string
    """

    def format_span(start, end):
        return f"{token_offsets[start][0]}..{token_offsets[end][1]}"

    spans = []
    last = None
    for tid in sorted(span_tokens):
        if last is None:
            start = end = tid
        elif tid == last + 1:
            end = tid
        else:
            spans.append(format_span(start, end))
            start = end = tid
        last = tid
    spans.append(format_span(start, end))
    return ";".join(spans)


def flat_tokens(tokens: List) -> List:
    return [token.text for token in tokens]


def output_file(output_dir: str, rels: List, doc_state, format: str = "tab") -> None:
    def format_label(lcased):
        parts = lcased.split(".")
        parts = [p[0].upper() + p[1:] for p in parts]
        return ".".join(parts)

    if not output_dir.endswith(os.sep):
        output_dir += os.sep

    rel_map = {}
    for rel in doc_state.rels.values():
        rel_map[rel.key] = rel

    token_offsets = {}
    if format in ["all","pdtb"]:
        # Make a mapping of tokens to character offsets
        cursor = 0
        for tok in doc_state.tokens:
            token_offsets[int(tok.doc_token_id)] = (cursor, cursor + len(tok.text))
            cursor += len(tok.text) + 1

    if format == "all":
        formats = ["tab", "rels", "pdtb"]
    else:
        formats = [format]
    for format in formats:
        if format == "rels":
            rows = ["\t".join(["doc","unit1_toks","unit2_toks","unit1_txt","unit2_txt","s1_toks","s2_toks","unit1_sent","unit2_sent","dir","rel_type","orig_label","label"])]
        elif format == "pdtb":
            rows = []
        else:
            rows = ["\t".join(['DOCNAME', 'TYPE', 'CONN', 'SENSE', 'RST', 'ARG1', 'ARG2', 'ARG1_IDS', 'ARG2_IDS', 'KEY', 'NOTES'])]
        for rel in rels:
            if format == "tab":
                rows.append("\t".join([str(x) for x in rel]))
                docname = rel[0]
            elif format == "pdtb":
                #rel_obj = rel_map[rel[9]]
                reltype = type_map[rel[1]]
                conn1 = rel[2]
                if "conn_tok_ids" in rel[10]:
                    conn_tokens = rel[10].split("conn_tok_ids=")[1].split(",")
                    conn_tokens = [int(tid) + 1 for tid in conn_tokens]
                    connspan = format_standoff(conn_tokens, token_offsets)
                else:
                    connspan = ""
                    conn_tokens = []
                arg1_edus = [doc_state.edus[e] for e in rel[7]]
                arg1_tokens = [tok+1 for edu in arg1_edus for tok in edu.tok_ids if tok+1 not in conn_tokens]
                arg1 = format_standoff(arg1_tokens, token_offsets)
                arg1_tokens = [t-1 for t in arg1_tokens]
                arg1_tokens = format_range(arg1_tokens)
                arg2_edus = [doc_state.edus[e] for e in rel[8]]
                arg2_tokens = [tok+1 for edu in arg2_edus for tok in edu.tok_ids if tok+1 not in conn_tokens]
                arg2 = format_standoff(arg2_tokens, token_offsets)
                arg2_tokens = [t-1 for t in arg2_tokens]
                arg2_tokens = format_range(arg2_tokens)
                if conn_tokens == []:
                    conn_tokens = ""
                else:
                    conn_tokens = [t-1 for t in conn_tokens]
                    conn_tokens = format_range(conn_tokens)
                row = [""] * 34
                row[0] = reltype
                row[1] = connspan
                row[7] = conn1
                row[8] = format_label(rel[3])
                row[14] = arg1
                row[20] = arg2
                row[31] = connspan if connspan != "" else arg2.split(";")[0].split("..")[0]
                if conn_tokens == "":
                    conn_tokens = "_"
                    start = arg2_tokens.split(",")[0].split("-")[0]
                else:
                    start = conn_tokens.split(",")[0].split("-")[0]
                row[32] = "GDTB::" + rel[0] + ":" + start + ";" + conn_tokens + ";" + arg1_tokens + ";" + arg2_tokens
                rows.append("|".join(row))
            else:
                docname, reltype, conn, sense, rst_rel, arg1, arg2, arg1_ids, arg2_ids, key, notes = rel
                if sense.lower() in ["norel","entrel"]:  # Don't export these in accordance with DISRPT shared task setup
                    continue
                direction = "1>2" if "arg1" in sense else "1<2"
                unit1_ids = arg1_ids if arg1_ids[0] < arg2_ids[0] else arg2_ids
                unit2_ids = arg2_ids if arg1_ids[0] < arg2_ids[0] else arg1_ids
                unit1_edus = [doc_state.edus[e] for e in unit1_ids]
                unit2_edus = [doc_state.edus[e] for e in unit2_ids]
                unit1_token_ids = [tid for edu in unit1_edus for tid in edu.tok_ids]
                unit2_token_ids = [tid for edu in unit2_edus for tid in edu.tok_ids]
                unit1_text = format_text(unit1_token_ids, doc_state.tokens)
                unit2_text = format_text(unit2_token_ids, doc_state.tokens)
                unit1_token_range = format_range(unit1_token_ids)
                unit2_token_range = format_range(unit2_token_ids)
                unit1_sents = [edu.sent_id for edu in unit1_edus]
                unit2_sents = [edu.sent_id for edu in unit2_edus]
                unit1_sents_tok_ids = [int(tok.doc_token_id) - 1 for tok in doc_state.tokens if tok.sent_id in unit1_sents]
                unit2_sents_tok_ids = [int(tok.doc_token_id) - 1 for tok in doc_state.tokens if tok.sent_id in unit2_sents]
                unit1_sents_text = format_text(unit1_sents_tok_ids, doc_state.tokens)
                unit2_sents_text = format_text(unit2_sents_tok_ids, doc_state.tokens)
                unit1_sents_tok_ids = format_range(unit1_sents_tok_ids)
                unit2_sents_tok_ids = format_range(unit2_sents_tok_ids)
                parts = sense.split(".")
                if len(parts) == 1:  # Match DISRPT format
                    modified_sense = sense.lower()
                elif len(parts) > 1:
                    modified_sense = parts[0].lower() + "." + parts[1].lower()
                row = [docname, unit1_token_range, unit2_token_range, unit1_text, unit2_text, unit1_sents_tok_ids, unit2_sents_tok_ids, unit1_sents_text, unit2_sents_text, direction, reltype, sense, modified_sense]
                rows.append("\t".join(row))

        if format == "pdtb":
            output_path = output_dir + "pdtb" + os.sep + "raw" + os.sep + "00" + os.sep
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path += docname
            raw_text = " ".join(flat_tokens(doc_state.tokens))
            with open(output_path, "w", encoding="utf8", newline="\n") as f:
                f.write(raw_text)
            output_path = output_dir + "pdtb" + os.sep + "gold" + os.sep + "00" + os.sep
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path += docname
            with open(output_path, "w", encoding="utf8", newline="\n") as f:
                f.write("\n".join(rows).strip() + "\n")
        else:
            output_path = output_dir + format + os.sep
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path += docname + "." + format
            with io.open(output_path, "w", encoding="utf8", newline="\n") as f:
                f.write("\n".join(rows).strip() + "\n")
