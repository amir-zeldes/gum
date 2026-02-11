from typing import Optional, Union, List
import re
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--merge", help="merge sentences",
                    action="store_true")
parser.add_argument("-s", "--split", help="split sentence",
                    action="store_true")
parser.add_argument("-i", "--input", help="input file")
parser.add_argument("-o", "--outfile", help="output file")
parser.add_argument("--sent_a", type=int, help="first sentence to merge")
parser.add_argument("--sent_b", type=int, help="second sentence to merge")
parser.add_argument("--sent_id", type=int, help="sent id for splitting")
parser.add_argument("--tok_id", type=int, help="tok id for splitting (this tok will start the new sent)")

args = parser.parse_args()

SENT_TEXT_RE = re.compile(r"#Text=(.*)")
TOKEN_RE = re.compile(r"^(\d+)-(\d+)\t")


# given a tsv file/tsv content and 2 sentence ids merge those two sentences and adjust the file, return tsv content or write to file (outfile if given or the input tsv)
def merge_sentences_webanno_tsv(
    tsv: Union[str, Path],
    sent_a: int,
    sent_b: int,
    outfile: Optional[Union[str, Path]] = None,
) -> str:
    """
    Merge sentence `sent_b` into `sent_a` in a WebAnno TSV 3.2 file.

    - `tsv`: path to a TSV file OR TSV content as a string
    - `sent_a`: first sentence id (kept)
    - `sent_b`: second sentence id (merged into sent_a)
    - `outfile`: optional output file path

    Returns merged TSV content as a string.
    """

    # ------------------------------------------------------------------
    # Load content
    # ------------------------------------------------------------------
    if isinstance(tsv, Path) or (isinstance(tsv, str) and Path(tsv).exists()):
        path = Path(tsv)
        lines = path.read_text().splitlines()
    else:
        lines = tsv.splitlines()

    # ------------------------------------------------------------------
    # Split into blocks: headers, sentences
    # ------------------------------------------------------------------
    header: List[str] = []
    sentences = {}  # sent_id -> list of lines
    sentence_texts = {}
    altered_tokens = {} # orig_tok -> altered_tok

    current_sent = 0
    for line in lines:
        if line.startswith("#Text="):
            current_sent += 1
            sentence_texts[current_sent] = line.split("#Text=")[1]
        elif current_sent == 0  and line.strip() != "": # header
            header.append(line)
        else:
            if current_sent not in sentences and line.strip() != "":
                sentences[current_sent] = [line]
            else:
                if "\t" in line: # no empty lines
                    sentences[current_sent].append(line)

    if sent_a not in sentences or sent_b not in sentences:
        raise ValueError("Both sentence IDs must exist in the TSV.")

    if sent_b != sent_a + 1:
        raise ValueError("This implementation assumes sent_b = sent_a + 1.")

    # ------------------------------------------------------------------
    # Merge #Text lines
    # ------------------------------------------------------------------

    new_sentence_texts = {}
    new_sentences = {}
    for sent_id in sentence_texts:
        if sent_id <= sent_a:
            new_sentence_texts[sent_id] = sentence_texts[sent_id]
        elif sent_id == sent_b:
            new_sentence_texts[sent_a] += " " + sentence_texts[sent_b]
        else:
            new_sentence_texts[sent_id - 1] = sentence_texts[sent_id]

    for sent_id in sentences:
        if sent_id <= sent_a:
            new_sentences[sent_id] = sentences[sent_id]
        elif sent_id == sent_b:
            new_sentences[sent_a] += sentences[sent_b]
        else:
            new_sentences[sent_id - 1] = sentences[sent_id]

    # ------------------------------------------------------------------
    # Merge tokens, renumbering token IDs (align sent ids)
    # ------------------------------------------------------------------

    for sent_id in new_sentences:
        # Make sure sent_id matches the sentence id in the line, if a token updated add to altered_tokens
        tok_id = 0
        for i, line in enumerate(new_sentences[sent_id]):
            tok_id += 1
            fields = line.split("\t")
            sent_tok_id = fields[0]
            sent_id_old = int(sent_tok_id.split("-")[0])
            if sent_id_old != sent_id:
                new_sent_tok_id = str(sent_id) + "-" + str(tok_id)
                fields[0] = new_sent_tok_id
                new_line = "\t".join(fields)
                new_sentences[sent_id][i] = new_line
                altered_tokens[sent_tok_id] = new_sent_tok_id

    # ------------------------------------------------------------------
    # Update edges
    # ------------------------------------------------------------------

    for sent_id in new_sentences:
        for i, line in enumerate(new_sentences[sent_id]):
            fields = line.split("\t")
            edges = fields[-2]
            if edges == "_":
                continue
            edges = edges.split("|")
            new_edges = []
            for edge in edges:
                start_token, remainder = edge.split("[")
                if start_token in altered_tokens:
                    new_edge = altered_tokens[start_token] + "[" + remainder
                    new_edges.append(new_edge)
                else:
                    new_edges.append(edge)
            fields[-2] = "|".join(new_edges)
            new_line = "\t".join(fields)
            new_sentences[sent_id][i] = new_line

    # ------------------------------------------------------------------
    # Reassemble TSV
    # ------------------------------------------------------------------
    output_lines = header
    output_lines.append("")
    for sent_id in new_sentence_texts:
        output_lines.append("")
        output_lines.append("#Text=" + new_sentence_texts[sent_id])
        for line in new_sentences[sent_id]:
            output_lines.append(line)
    output_lines.append("")

    output = "\n".join(output_lines)

    # ------------------------------------------------------------------
    # Write output if requested
    # ------------------------------------------------------------------
    if outfile:
        Path(outfile).write_text(output)
    elif isinstance(tsv, (str, Path)) and Path(tsv).exists():
        Path(tsv).write_text(output)

    return output


# given a tsv file/tsv content and a sent-tok id, split the sentence at that token (that token starts new sent), (outfile if given or the input tsv)
def split_sentence_webanno_tsv(
    tsv: Union[str, Path],
    sent_id: int,
    tok_id: int,
    outfile: Optional[Union[str, Path]] = None,
) -> str:
    """
    Split sentence `sent_id` at token `tok_id`.
    The token `sent_id-tok_id` becomes the FIRST token of the new sentence.

    - `tsv`: path to TSV file or TSV content string
    - `sent_id`: sentence id to split
    - `tok_id`: token id where new sentence starts
    - `outfile`: optional output file

    Returns updated TSV content as a string.
    """

    # ------------------------------------------------------------------
    # Load content
    # ------------------------------------------------------------------
    if isinstance(tsv, Path) or (isinstance(tsv, str) and Path(tsv).exists()):
        path = Path(tsv)
        lines = path.read_text().splitlines()
    else:
        lines = tsv.splitlines()

    # ------------------------------------------------------------------
    # Parse header + sentence blocks
    # ------------------------------------------------------------------
    header: List[str] = []
    sentences = {}  # sent_id -> list of lines
    sentence_texts = {}
    altered_tokens = {} # orig_tok -> altered_tok

    current_sent = 0
    for line in lines:
        if line.startswith("#Text="):
            current_sent += 1
            sentence_texts[current_sent] = line.split("#Text=")[1]
        elif current_sent == 0  and line.strip() != "": # header
            header.append(line)
        else:
            if current_sent not in sentences and line.strip() != "":
                sentences[current_sent] = [line]
            else:
                if "\t" in line: # no empty lines
                    sentences[current_sent].append(line)

    if sent_id not in sentences or tok_id - 1 not in range(len(sentences[sent_id])):
        raise ValueError("Both sentence ID and tok ID must exist in the TSV.")

    # ------------------------------------------------------------------
    # Split text and tokens
    # ------------------------------------------------------------------

    new_sentence_texts = {}
    new_sentences = {}
    for curr_sent_id in sentence_texts:
        if curr_sent_id < sent_id:
            new_sentence_texts[curr_sent_id] = sentence_texts[curr_sent_id]
        elif curr_sent_id == sent_id:
            tokens = sentence_texts[curr_sent_id].split(" ")
            part1 = " ".join(tokens[:tok_id - 1])
            part2 = " ".join(tokens[tok_id - 1:])
            new_sentence_texts[curr_sent_id] = part1
            new_sentence_texts[curr_sent_id + 1] = part2
        else:
            new_sentence_texts[curr_sent_id + 1] = sentence_texts[curr_sent_id]

    for curr_sent_id in sentences:
        if curr_sent_id < sent_id:
            new_sentences[curr_sent_id] = sentences[curr_sent_id]
        elif curr_sent_id == sent_id:
            new_sentences[curr_sent_id] = sentences[curr_sent_id][:tok_id - 1]
            new_sentences[curr_sent_id + 1] = sentences[curr_sent_id][tok_id - 1:]
        else:
            new_sentences[curr_sent_id + 1] = sentences[curr_sent_id]

    # ------------------------------------------------------------------
    # Update IDs
    # ------------------------------------------------------------------

    for curr_sent_id in new_sentences:
        # Make sure curr_sent_id matches the sentence id in the line, if a token updated add to altered_tokens
        tok_id = 0
        for i, line in enumerate(new_sentences[curr_sent_id]):
            tok_id += 1
            fields = line.split("\t")
            sent_tok_id = fields[0]
            sent_id_old = int(sent_tok_id.split("-")[0])
            if sent_id_old != curr_sent_id:
                new_sent_tok_id = str(curr_sent_id) + "-" + str(tok_id)
                fields[0] = new_sent_tok_id
                new_line = "\t".join(fields)
                new_sentences[curr_sent_id][i] = new_line
                altered_tokens[sent_tok_id] = new_sent_tok_id

    # ------------------------------------------------------------------
    # Update edges
    # ------------------------------------------------------------------

    for curr_sent_id in new_sentences:
        for i, line in enumerate(new_sentences[curr_sent_id]):
            fields = line.split("\t")
            edges = fields[-2]
            if edges == "_":
                continue
            edges = edges.split("|")
            new_edges = []
            for edge in edges:
                start_token, remainder = edge.split("[")
                if start_token in altered_tokens:
                    new_edge = altered_tokens[start_token] + "[" + remainder
                    new_edges.append(new_edge)
                else:
                    new_edges.append(edge)
            fields[-2] = "|".join(new_edges)
            new_line = "\t".join(fields)
            new_sentences[curr_sent_id][i] = new_line

    # ------------------------------------------------------------------
    # Reassemble TSV
    # ------------------------------------------------------------------
    
    output_lines = header
    output_lines.append("")
    for curr_sent_id in new_sentence_texts:
        output_lines.append("")
        output_lines.append("#Text=" + new_sentence_texts[curr_sent_id])
        for line in new_sentences[curr_sent_id]:
            output_lines.append(line)
    output_lines.append("")

    output = "\n".join(output_lines)

    # ------------------------------------------------------------------
    # Write output if requested
    # ------------------------------------------------------------------
    if outfile:
        Path(outfile).write_text(output)
    elif isinstance(tsv, (str, Path)) and Path(tsv).exists():
        Path(tsv).write_text(output)

    return output


if __name__ == "__main__":
    if args.merge:
        merge_sentences_webanno_tsv(
            args.input,
            args.sent_a,
            args.sent_b,
            args.outfile
        )

    if args.split:
        split_sentence_webanno_tsv(
            args.input,
            args.sent_id,
            args.tok_id,
            args.outfile
        )

