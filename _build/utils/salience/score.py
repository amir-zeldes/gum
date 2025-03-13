import os
import glob
from repair_tsv import adjust_edges, fix_genitive_s


def get_sal_tsv(input_paths): #get salient entities (only first mentions)
    all_results = []

    def extract_bracketed_number(s):
        start = s.find("[")
        end = s.find("]")
        if start != -1 and end != -1:
            return s[start+1:end]
        return None

    def remove_bracketed_number(s):
        parts = s.split('|')  # Split by "|" to handle multiple coref indices
        cleaned_parts = [part.split('[')[0] for part in parts]  # Remove everything after "["
        return ','.join(cleaned_parts)  # Rejoin the cleaned parts

    # Check if input_paths is a directory or a list of file paths
    if isinstance(input_paths, str) and os.path.isdir(input_paths):
        filepaths = sorted(glob.glob(os.path.join(input_paths, "*.tsv")))
    elif isinstance(input_paths, list):
        # Prepend './data/input/tsv/' and append '.tsv' to each file name in the list
        filepaths = [os.path.join('./data/input/tsv', f"{filename}.tsv") for filename in input_paths]
    else:
        raise ValueError("input_paths must be a directory or a list of file paths")

    for filepath in filepaths:
        # Convert to absolute path to ensure it's correct
        filepath = os.path.abspath(filepath)
        
        file_result = []
        try:
            with open(filepath, 'r') as file:
                word_dict = {}
                word_indices = {}
                coref_indices = {}
                
                for line in file:
                    columns = line.strip().split('\t')
                    if len(columns) < 7:
                        continue

                    word_index = columns[0]
                    word = columns[2]
                    ent_type = columns[3].split('|')  # Extract ent_type values
                    col5_values = columns[4].split('|')
                    col6_values = columns[5].split('|')
                    coref_index = columns[-1]  
                    
                    for col5, col6 in zip(col5_values, col6_values):
                        if col6.startswith('sal') and not col5.startswith('giv'):
                            sal_number = extract_bracketed_number(col6)
                            if sal_number:
                                if sal_number not in word_dict:
                                    word_dict[sal_number] = []
                                    word_indices[sal_number] = []
                                    coref_indices[sal_number] = []
                                word_dict[sal_number].append(word)
                                word_indices[sal_number].append(word_index)
                                coref_indices[sal_number].append(coref_index)

                for key in word_dict:
                    concatenated_words = " ".join(word_dict[key])
                    concatenated_indices = ", ".join(word_indices[key])
                    # Remove bracketed numbers and concatenate
                    filtered_corefs = [remove_bracketed_number(coref) for coref in coref_indices[key] if coref and coref != "_"]
                    concatenated_corefs = ", ".join(filtered_corefs)
                    file_result.append((concatenated_words, concatenated_indices, concatenated_corefs))
        
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            continue  # Optionally, skip to the next file if an error occurs
        
        all_results.append(file_result)
    
    return all_results

def get_sal_mentions(input_paths, gum_target=False):
    # get all salient mentions, only works in gum_target, else conversion must be done on the fly
    all_results = {}

    def extract_bracketed_number(s):
        start = s.find("[")
        end = s.find("]")
        if start != -1 and end != -1:
            return s[start+1:end]
        return None

    def remove_bracketed_number(s):
        parts = s.split('|')  # Split by "|" to handle multiple coref indices
        cleaned_parts = [part.split('[')[0] for part in parts]  # Remove everything after "["
        return ','.join(cleaned_parts)  # Rejoin the cleaned parts

    # Check if input_paths is a directory or a list of file names
    if isinstance(input_paths, str) and os.path.isdir(input_paths):
        filepaths = sorted(glob.glob(os.path.join(input_paths, "*.tsv")))
    elif isinstance(input_paths, list):
        # Prepend './data/input/tsv/' and append '.tsv' to each file name in the list
        filepaths = [os.path.join('./data/input/tsv', f"{filename}.tsv") for filename in input_paths]
    else:
        raise ValueError("input_paths must be a directory or a list of file paths")

    for filepath in filepaths:
        # Convert to absolute path to ensure it's correct
        filepath = os.path.abspath(filepath)
        docname = os.path.basename(filepath).replace('.tsv', '')
        
        file_result = []
        tt_file = filepath.replace('tsv', 'xml')
        if gum_target:
            # This is faster but requires a full build of GUM to run first
            filepath = filepath.replace('src', 'target' + os.sep + 'coref')

        with open(filepath, 'r', encoding="utf8") as file:
            in_tsv = file.read()
            if not gum_target:
                parsed_lines, entity_mappings, single_tok_mappings = fix_genitive_s(in_tsv, tt_file, warn_only=True, string_input=True)
                propagated, _, _, _ = adjust_edges(in_tsv, parsed_lines, entity_mappings, single_tok_mappings)
            else:
                propagated = in_tsv
            lines = propagated.split('\n')
            word_dict = {}
            word_indices = {}
            coref_indices = {}
            
            for line in lines:
                columns = line.strip().split('\t')
                if len(columns) < 7:
                    continue
                    
                word_index = columns[0]
                word = columns[2]
                col5_values = columns[4].split('|')
                col6_values = columns[5].split('|')
                coref_index = columns[-1]  
                
                for col5, col6 in zip(col5_values, col6_values):
                    if col6.startswith('sal'):
                        sal_number = extract_bracketed_number(col6)
                        if sal_number:
                            if sal_number not in word_dict:
                                word_dict[sal_number] = []
                                word_indices[sal_number] = []
                                coref_indices[sal_number] = []
                            word_dict[sal_number].append(word)
                            word_indices[sal_number].append(word_index)
                            coref_indices[sal_number].append(coref_index)

            for key in word_dict:
                concatenated_words = " ".join(word_dict[key])
                concatenated_indices = ", ".join(word_indices[key])
                # Remove bracketed numbers and concatenate
                filtered_corefs = [remove_bracketed_number(coref) for coref in coref_indices[key] if coref and coref != "_"]
                concatenated_corefs = ", ".join(filtered_corefs)
                file_result.append((concatenated_words, concatenated_indices, concatenated_corefs))
        
        all_results[docname] = file_result
    
    return all_results


def sal_coref_cluster(sal_mentions):
    def find_coref_chain(start_tuple):
        chain = [start_tuple[0]]  # Start with the word span of the first tuple
        current_word_index = start_tuple[1].split(',')[0].strip()  # Only use the first word index
        current_coref_indices = [ci.strip() for ci in start_tuple[2].split(',')]

        while True:
            found = False
            for tup in file_result:
                if tup in used_indices:
                    continue

                words, word_indices, coref_indices = tup
                first_word_index = word_indices.split(',')[0].strip()  # Only check the first index
                coref_indices_list = [ci.strip() for ci in coref_indices.split(',')]

                if first_word_index in current_coref_indices:
                    chain.append(words)
                    used_indices.add(tup)
                    current_word_index = first_word_index
                    current_coref_indices = coref_indices_list
                    found = True
                    break

            if not found:
                break

        return tuple(chain)

    sal_coref_clusters = {}

    # Define a set of pronouns (both upper and lowercase)
    pronouns = {'he', 'she', 'it', 'they', 'we', 'i', 'you', 
                'him', 'her', 'them', 'us', 'me', 'it', 'there',
                'his', 'hers', 'its', 'their', 'our', 'my', 'your',
                'He', 'She', 'It', 'They', 'We', 'I', 'You', 
                'Him', 'Her', 'Them', 'Us', 'Me', 'It', 'There',
                'His', 'Hers', 'Its', 'Their', 'Our', 'My', 'Your'}

    for docname, file_result in sal_mentions.items():
        cluster = []
        used_indices = set()

        for tup in file_result:
            words, word_indices, coref_indices = tup

            # Handle singletons by only including the word span as a single string
            if coref_indices == "":
                # Check if the word is a pronoun, and if so, skip this tuple
                if words not in pronouns:
                    cluster.append((words,))
                continue

            if tup not in used_indices:
                coref_chain = find_coref_chain(tup)
                if len(coref_chain) > 1:
                    cluster.append(coref_chain)
                used_indices.add(tup)

        sal_coref_clusters[docname] = cluster

    return sal_coref_clusters


def extract_first_mentions(sc, sum1_alignments, summaries, exact=False):
    results = {}

    for doc_index, docname in enumerate(sc):
        st_doc = sum1_alignments[docname]
        sc_doc = sc[docname]
        doc_summaries = summaries[docname]
        sources = list(doc_summaries.keys())
        doc_results = {}
        for summary_index, alignment_list in enumerate(st_doc):
            source = sources[summary_index].split("/")[-1]#.replace("; postedited", "")
            seen_mentions = set()  # Set to keep track of unique mentions
            doc_results[source] = []
            # If alignment_list is empty, continue with an empty list in the results
            if not alignment_list:
                continue

            for alignment in sorted(alignment_list,key=lambda x: len(str(x[0])),reverse=True):
                salient_mention = alignment[0].strip().lower()  # Get the salient mention in lowercase

                for sc_tuple in sorted(sc_doc,key=lambda x: len(str(x))):
                    sc_mentions = [mention.strip().lower() for mention in sc_tuple]  # Normalize mentions in sc to lowercase

                    # Check if the salient mention is a substring of any mention in sc
                    if exact:
                        if any(salient_mention == mention for mention in sc_mentions) and \
                                sc_tuple[0] not in seen_mentions:
                            if not any([sc_tuple[0] == x[0] for x in doc_results[source]]):
                                doc_results[source].append((sc_tuple[0],sc_tuple[1].split(",")[0]))  # Append the first matching mention from sc
                            seen_mentions.add(sc_tuple[0].lower())  # Mark this mention as seen
                            break  # Break after finding the match for this alignment
                    else:
                        if any(" " + salient_mention + " " in " " + mention + " " for mention in sc_mentions) and sc_tuple[0] not in seen_mentions:
                            if sc_tuple[0] not in doc_results[source]:
                                doc_results[source].append(sc_tuple[0]) # Append the first matching mention from sc
                            seen_mentions.add(sc_tuple[0].lower())  # Mark this mention as seen
                            break  # Break after finding the match for this alignment

        if exact:
            for source in doc_results:
                doc_results[source] = [mention[0] for mention in sorted(doc_results[source], key=lambda x: (int(x[1].split("-")[0]),int(x[1].split("-")[1])))]

        results[docname] = doc_results

    return results


def calculate_scores(pred, gold):
    # Initialize variables for micro-average calculation
    total_matches = 0
    total_pred_mentions = 0
    total_gold_mentions = 0

    # Initialize lists for macro-average and per-document scores
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # List to hold scores for each document individually
    per_document_scores = []

    # Loop over each document's predictions and gold standard
    for pred_doc, gold_doc in zip(pred, gold):
        pred_mentions = [p for p in pred_doc if p is not None]  # Filter out None values
        gold_mentions = [g[0] for g in gold_doc]  # Extract the first element (word span) from gold tuples

        # Update micro-average counters
        total_pred_mentions += len(pred_mentions)
        total_gold_mentions += len(gold_mentions)

        # Count matches between predicted mentions and gold mentions
        matches = sum(1 for pm in pred_mentions if pm in gold_mentions)
        total_matches += matches

        # Calculate precision, recall, and F1 for this document (macro calculation)
        precision = matches / len(pred_mentions) if len(pred_mentions) > 0 else 0
        recall = matches / len(gold_mentions) if len(gold_mentions) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Append the scores for this document to the lists
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

        # Store per-document scores in a dictionary
        per_document_scores.append({
            'precision': precision,
            'recall': recall,
            'f1': f1_score
        })

    # Macro-average calculation (average over documents)
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    # Micro-average calculation (aggregate totals across all documents)
    micro_precision = total_matches / total_pred_mentions if total_pred_mentions > 0 else 0
    micro_recall = total_matches / total_gold_mentions if total_gold_mentions > 0 else 0
    micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Return both macro, micro averages, and individual document scores
    return {
        'macro_precision': avg_precision,
        'macro_recall': avg_recall,
        'macro_f1': avg_f1_score,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1_score,
        'per_document_scores': per_document_scores
    }
