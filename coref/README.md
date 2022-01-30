# Coreference and entity annotations

This directory contains release versions of GUM's entity and coreference annotations. As with other directories, the compiled data for the Reddit subcorpus is not included and must be reconstructed using the build bot (see [../README_reddit.md](README_reddit.md). This directory includes:

  * gum/ - the native GUM scheme coreference and entity annotations in multiple formats
  * ontogum/ - the converted coreference annotations following the OntoNotes coreference scheme
  * wiki_map.tab - a mapping of GUM's wikification identifiers based on Wikipedia links and article titles to Wikidata identifiers (can be recomputed using `_build/utils/wiki_identifier.py`)
