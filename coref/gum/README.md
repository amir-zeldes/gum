# GUM coreference data

This directory contains the main version of the coreference annotations in GUM, following GUM's own annotation scheme. This means in particular:

  * Singleton mentions are included (entities mentioned only once in a document)
  * Non-named compound modifiers may corefer with other mentions and are therefore included
  * Predicative mentions may be considered (co-)referential and are included, unless they are non-referring
  * Indefinite NPs can be antecedents and anaphors
  * Co-referetial verbal spans (e.g. "X [visited Spain] ... the visit") are based on meaning, and may include the entire VP, an entire sentence, or other spans as semantically appropriate
  * Files in tsv/ also include bridging anaphora and separate coreference types for appositions, cataphora, discourse deixis, predicative coreference and pronominal anaphora

If you are looking for more restricted coreference annotations following the OntoNotes scheme, please see the `ontogum/` directory. If you are looking for GUM annotations in the CoNLL-U format used for the Universal Anaphora project (https://universalanaphora.github.io/UniversalAnaphora/), please see the top level dep/ directory.
