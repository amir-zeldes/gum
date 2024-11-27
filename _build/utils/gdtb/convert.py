from typing import Dict
from nodes import Doc, Relation


class Converter:
    def __init__(self, doc: Doc, modules: Dict):
        self.doc = doc
        self.modules = modules

    def convert(self, rel: Relation, cache: str = "full"):
        """
        Iterate each conversion function
        """
        # remove non-pdtb relations, including EDUs that are PP or NPs in parentheses
        if len(rel.source.edus) == 1:
            head_edu = rel.source.edus[0]
            if head_edu.head_func in ["obl","appos","nmod"]:  # Remove PP and NP in parentheses or strong DM preposition
                return

        for module_name, module in self.modules.items():
            if cache == "none" and module_name == "cache":
                continue  # For testing full auto mode
            module.convert(self.doc, rel)

            # Check for "and (then)" explicit+implicit sequence
            if module_name == "explicit" and "sequence" in rel.relname:
                if len(rel.pdtb_rels["implicit"]) == 0:  # Ensure no cached implicit relation
                    dm_sigs = [sig for sig in rel.signals if sig.subtype in ["dm","orphan"]]
                    if len(dm_sigs) == 1:
                        if dm_sigs[0].toks[0].text.lower() in ["and","also"]:  # Explicit sequence marked by non-temporal DM
                            # add implicit 'then'
                            rel.pdtb_rels["cache"].append(("implicit", "temporal.asynchronous.precedence", [], "then"))

            # Check for "by" explicit manner+cause/condition/purpose
            if module_name == "explicit" and "mode" in rel.relname:
                if len(rel.pdtb_rels["explicit"]) == 1:  # Ensure we already have the regular manner relation
                    dm_sigs = [sig for sig in rel.signals if sig.subtype in ["dm","orphan"]]
                    if len(dm_sigs) == 1:
                        if dm_sigs[0].toks[0].text.lower() == "by":  # Explicit manner usually gets secondary sense
                            # add second explicit 'by' relation
                            probas = module.get_rel_probs(self.doc.docname, rel, self.doc.sents)
                            probas = sorted(probas, key=lambda x: probas[x], reverse=True)
                            for sense in probas:
                                if any([x in sense for x in ["cause","condition","purpose"]]):
                                    break
                            rel.pdtb_rels["cache"].append(("explicit", sense, dm_sigs[0].tok_ids, "by"))

            