# gum
Repository for the Georgetown University Multilayer Corpus (GUM)

This repository contains release versions of the Georgetown University Multilayer Corpus (GUM), a corpus of English texts from four text types (interviews, news, travel guides, instructional texts). The corpus is created as part of the course LING-367 (Computational Corpus Linguistics) at Georgetown University. For more details see: http://corpling.uis.georgetown.edu/gum.

## Directories
The corpus is downloadable in multiple formats. Not all formats contain all annotations. The most complete XML representation is in PAULA XML, and the easiest way to search in the corpus is using ANNIS. Other formats may be useful for other purposes. See website for more details.

  * annis/ - The entire merged corpus, with all annotations, as a relANNIS 3.3 corpus dump, importable into ANNIS (see http://corpus-tools.org/annis)
  * const/ - Constituent trees and PTB POS tags in the PTB bracketing format (automatic parser output)
  * coref/ - Entity and coreference annotation in two formats: conll shared task tabular format (with no bridging annotations) and OntoNotes .coref format, including entity and information status annotations, bridging and singleton entities
  * dep/ - Dependency trees in the conll 10 column format using Stanford Typed Dependencies (manually corrected) and extended PTB POS tags (following TreeTagger/Amalgam, e.g. tags like VVZ)
  * paula/ - The entire merged corpus in PAULA standoff XML, with all annotations (see https://www.sfb632.uni-potsdam.de/en/paula.html for format documentation) 
  * rst/ - Rhetorical Structure Theory analyses in .rs3 format as used by RSTTool and rstWeb (spaces between words correspond to tokenization in rest of corpus)
  * xls/ - A giant spreadsheet containing all POS tags (vanilla and extended PTB, claws5), structural TEI annotations (including rough speech act), dependencies and constituent trees, but not the coreference, entity or RST annotations
  * xml/ - vertical XML representations with 1 token or tag per line and tab delimited lemmas and POS tags (extended, VVZ style), compatible with the IMS Corpus Workbench (a.k.a. TreeTagger format).