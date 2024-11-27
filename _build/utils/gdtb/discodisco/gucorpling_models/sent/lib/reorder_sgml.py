import io, re
from collections import defaultdict

PRIORITIES = [
    "text",
    "sp",
    "table",
    "row",
    "cell",
    "head",
    "p",
    "figure",
    "caption",
    "list",
    "item",
    "s",
    "quote",
    "q",
    "hi",
    "sic",
    "ref",
    "date",
    "incident",
    "w",
]

OPEN_SGML_ELT = re.compile(r"^<([^/ ]+)( .*)?>$")
CLOSE_SGML_ELT = re.compile(r"^</([^/]+)>$")


class Span:
    def __init__(self, start=0, end=0, text="", elem="", priorities=PRIORITIES):
        self.start = start
        self.end = end
        self.text = text
        self.elem = elem
        self.length = end - start
        self.priority = priorities.index(elem) if elem in priorities else 100

    def __repr__(self):
        return str(self.start) + "-" + str(self.end) + ": " + self.text


def reorder(tt_sgml, priorities=PRIORITIES):
    # Preprocessing: if an element opens and closes immediately, the rest of the code will handle it incorrectly
    # unless we add a dummy token. It will be removed later.
    lines = tt_sgml.split("\n")
    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        next_line = lines[i + 1]
        open_match = re.search(OPEN_SGML_ELT, line)
        close_match = re.search(CLOSE_SGML_ELT, next_line)
        if open_match and close_match and open_match.groups()[0] == close_match.groups()[0]:
            lines.insert(i + 1, "__SECRET_DUMMY_TOKEN__")
            i += 1
        i += 1

    # Pass 1: build data model
    open_elems = defaultdict(list)
    spans = []
    toknum = 1
    for line in lines:
        print(line)
        if line.startswith("</") and line.endswith(">"):  # Close element
            elem = re.search(r"^</([^\s>]*)", line).group(1)
            if elem not in open_elems:
                import pdb; pdb.set_trace();
                raise IOError("! saw a closed element: " + line + " but no corresponding element is open!\n")
            span = Span(
                start=open_elems[elem][-1][0],
                end=toknum,
                text=open_elems[elem][-1][1],
                elem=elem,
                priorities=priorities,
            )
            open_elems[elem].pop()
            if len(open_elems[elem]) == 0:
                del open_elems[elem]
            spans.append(span)
        elif (line.startswith("<") and line.endswith("/>")) or line.startswith("<?"):  # Unary element, treat like token
            toknum += 1
        elif line.startswith("<") and line.endswith(">"):  # Open element
            elem = re.search(r"^<([^\s>]*)", line).group(1)
            open_elems[elem].append((toknum, line))
        elif len(line.strip()) > 0:
            toknum += 1

    # Build start/end dictionaries
    start_dict = defaultdict(list)
    end_dict = defaultdict(list)
    for span in spans:
        start_dict[span.start].append(span)
        end_dict[span.end].append(span)

    # Pass 2: reorder
    output = []
    toknum = 1
    for line in lines:
        if ((line.startswith("<") and line.endswith(">")) or len(line.strip()) == 0) and not (
            line.startswith("<?") or (line.startswith("<") and line.endswith("/>"))
        ):
            continue

        starting = start_dict[toknum]
        elems = sorted(starting, key=lambda x: (-x.length, x.priority))
        for elem in elems:
            output.append(elem.text)

        output.append(line)
        toknum += 1

        ending = end_dict[toknum]
        elems = sorted(ending, key=lambda x: (x.length, -x.priority))
        for elem in elems:
            output.append("</" + elem.elem + ">")

    return "\n".join(output).replace("__SECRET_DUMMY_TOKEN__\n", "") + "\n"


if __name__ == "__main__":
    test_sgml = """<?xml version="1.0" ?>
<text dateCollected="2019-11-05" dateCreated="2019-02-04" dateModified="2019-04-23" id="autogum_bio_doc031" shortTile="ingo-ruczinski" sourceURL="https://en.wikipedia.org/wiki/Ingo_Ruczinski" speakerCount="0" speakerList="none" title="Ingo Ruczinski" type="bio">
<head>
<s>
Awards
and
honors
</s>
</head>
<p>
<s>
In
2016
</s>
<s>
Ingo
Ruczinski
had
became
an
elected
fellow
of
the
<ref target="https://en.wikipedia.org/wiki/American_Statistical_Association">
American
Statistical
Association
</ref>
.
<ref target="https://en.wikipedia.org/wiki/Main_Page#cite_note-2">
[
2
]
</s>
</ref>
</p>
</text>"""

    print(reorder(test_sgml))
