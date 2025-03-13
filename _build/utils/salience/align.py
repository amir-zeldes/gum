import os
import sys
import argparse
from openai import OpenAI
from transformers import pipeline
import random
from collections import defaultdict
from get_summary import get_summary, get_summary_gpt4o, get_summary_claude35, extract_gold_summaries_from_xml, extract_text_speaker_from_xml, read_documents
from apis import gpt4_key

client = OpenAI(api_key=gpt4_key)

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
gum_src = script_dir + ".." + os.sep + ".." + os.sep + "src" + os.sep


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

def replace_empty_strings(data):
    # Replace empty strings with "_"
    if isinstance(data, list):
        # Recursively process each item in the list
        return [replace_empty_strings(item) for item in data]
    elif isinstance(data, tuple):
        # Convert tuple to list, process, and convert back to tuple
        return tuple(replace_empty_strings(item) for item in data)
    elif isinstance(data, str):
        return "_" if data == "" else data
    else:
        # Return data as is if not list, tuple, or string
        return data

def extract_mentions_from_gold_tsv(data_folder, docnames=None):
    all_mentions = []

    # List all TSV files in the folder
    tsv_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".tsv")])

    for tsv_file in tsv_files:
        file_path = os.path.join(data_folder, tsv_file)
        docname = os.path.basename(tsv_file).split(".")[0]
        if docnames is not None:
            if docname not in docnames:
                continue
        mentions = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        current_mentions = {}

        for line in lines:
            if line.startswith("#") or not line.strip():
                continue

            columns = line.strip().split("\t")
            word_index = columns[0]
            word = columns[2]
            entity_type = columns[3]
            coref_info = columns[-1]
            
            # Handle nested mentions
            entity_types = entity_type.split("|")

            # Track current mentions for each entity type
            for i, entity in enumerate(entity_types):
                if entity == "_" or not entity:
                    continue

                if entity not in current_mentions:
                    current_mentions[entity] = ([], [], coref_info)

                current_mentions[entity][0].append(word)
                current_mentions[entity][1].append(word_index)

            # Check for completion of current mentions
            completed_mentions = []
            for entity in list(current_mentions.keys()):
                if entity not in entity_types:
                    completed_mentions.append(entity)

            # Add completed mentions to the result
            for entity in completed_mentions:
                word_span, indices, coref_index = current_mentions.pop(entity)
                mentions.append((" ".join(word_span), ",".join(indices), coref_index))

        # Add any remaining mention
        for entity, (word_span, indices, coref_index) in current_mentions.items():
            mentions.append((" ".join(word_span), ",".join(indices), coref_index))

        all_mentions.append(mentions)

    return all_mentions


def get_entities_from_gold_tsv(data_folder, docnames=None, first=False):
    all_results = []
    tsv_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".tsv")])

    for tsv_file in tsv_files:
        filepath = os.path.join(data_folder, tsv_file)
        docname = os.path.basename(tsv_file).split(".")[0]
        if docnames is not None:
            if docname not in docnames:
                continue
        
        file_result = []
        with open(filepath, 'r') as file:
            word_dict = {}
            word_indices = {}
            coref_indices = {}
            etype_dict = {}
            
            for line in file:
                columns = line.strip().split('\t')
                if len(columns) < 7:
                    continue
                
                word_index = columns[0]
                word = columns[2]
                col4_values = columns[3].split('|')
                col5_values = columns[4].split('|')
                col6_values = columns[5].split('|')
                coref_index = columns[-1]
                
                # Now only extract mentions that are first mentions 
                for i, tup in enumerate(zip(col5_values, col6_values)):
                    col5, col6 = tup
                    if not first or not col5.startswith('giv'):  # First mentions or all mentions
                        cls_number = extract_bracketed_number(col5)
                        if cls_number:
                            if cls_number not in word_dict:
                                word_dict[cls_number] = []
                                word_indices[cls_number] = []
                                coref_indices[cls_number] = []
                                etype_dict[cls_number] = []
                            word_dict[cls_number].append(word)
                            word_indices[cls_number].append(word_index)
                            coref_indices[cls_number].append(coref_index)
                            etype_dict[cls_number].append(col4_values[i])

            for key in word_dict:
                concatenated_words = " ".join(word_dict[key])
                concatenated_indices = ", ".join(word_indices[key])
                # Remove bracketed numbers and concatenate
                filtered_corefs = [remove_bracketed_number(coref) for coref in coref_indices[key] if coref and coref != "_"]
                concatenated_corefs = ", ".join(filtered_corefs)
                concatenated_entity_types = " ".join(etype_dict[key])
                #file_result.append((concatenated_words, concatenated_indices, concatenated_corefs))
                file_result.append((concatenated_words, concatenated_indices, concatenated_entity_types))

        all_results.append(file_result)
    
    return all_results

def align_llm(doc_mentions, summary_text, doc_text, use_cached_prompts=True, style=1, gpt="gpt-4o-mini"):
    """
    Align mentions using GPT-4o API (chat model) with custom parameters for temperature and top_p.

    Args:
        doc_mentions (dict of list of tuples): dictionary from docname to lists of tuples where each tuple contains (word_span, word_index, coref_index).
        summary_text (dict of list of str): dictionary from docname to lists of summaries.
        doc_text (dict of str): dictionary of document texts (one for each summary).
        use_cached_prompts (bool): Whether to use cached prompts for different genres.
        style (int): Style of the prompt to use (1 or 2).
        gpt (str): Model to use for alignment.

    Returns:
        dict of list of list of tuples: A dictionary of document name to lists of lists of tuples where each tuple's `word_span` is found in the corresponding document.
    """

    if not live:
        sys.stderr.write("o LLM: running in simulation mode!\n")

    prompt_template = (
        "For each of the entities mentioned in a document, please "
        "return the exact same entity as it appears in the phrase listed under 'Entities', if and only if it is mentioned in the summary. Otherwise, don't return anything and move on to the next. "
        "When matching, please also consider synonyms or alternative phrases that refer to the same entity. If a speaker says 'I' or is mentioned as 'you', then the speaker's name or label is considered mentioned (e.g. Kim)\n\n"
        "For example:\n\n"
        "Example document:\nJennifer: We need a —  Jennifer: Do you have any sharp objects on you ? Dan: No . Dan: Keys ? Jennifer: No I need like a little pin or something . Jennifer: You have a pencil ? Dan: You have anything in your hair ? Jennifer: No . Jennifer: Fuck . Dan: What do you have to hit ? Jennifer: See this is the little -  Jennifer: Oh . Jennifer: Oh oh . Dan: Cool ? Jennifer: Okay . Jennifer: Alright . Jennifer: See , it was just slow . Jennifer: Okay . Jennifer: This is me ? Jennifer: Is this me ? Dan: Yeah . Dan: Yeah . Dan: Jennifer . Jennifer: Oh . Jennifer: That 's right . Dan: There you go thinking again . Jennifer: Smart ass . Jennifer: Smart ass . Jennifer: Alright . Dan: Wow . Dan: Who took over uh ... Jennifer: Oh . Jennifer: They got North America . Jennifer: But not for long . Jennifer: Oh , my God . Jennifer:  Oh my God , did you see that ? Dan: Because player thr- player three is aggressive , so he 's gon na like go for everything . Jennifer: How do you know ? Jennifer: Did I make him aggressive ? Dan: Yeah , you made him aggressive , so , he 's gon na like , try to tear everything up now . Dan: Um , that 's pretty well , like secure right there , so maybe —  Dan: That 's me . Jennifer: Oh fuck . Dan: Wow , he wiped my ass out . Jennifer: Ah , you suck . Jennifer: Watch this . Jennifer: Loser . Jennifer: What else can we do tomorrow ? Jennifer: Besides go to the movies , t- ? Dan: Go out to dinner ? Jennifer: I 'm so not hungry right now , it 's hard for me to think about food . Dan: Alright . Jennifer: I 'd like to go out to dinner though . Jennifer: Think we can find a hot dog ? Dan: Yeah , that 's a good idea . Dan: That 's an excellent idea . Jennifer: There you go thinking again again . Dan: There you go thinking again . Jennifer: I 'm gon na whip your butt . Dan: You think so , hunh ? Jennifer: Yeah . Dan: Un-unh . Dan: That 's all I get ? Dan: That 's me , right ? Jennifer: Yeah you get a percentage of the amount of countries you own , and then , for continents you get another set amount . Dan: So can I get something on this bad boy ? Jennifer: Yeah . Jennifer: See ? Dan: So I hit okay ? Jennifer: Yeah . Jennifer: Hit okay . Jennifer: See you got one of each kind of card . Dan: Excellent . Dan: Oh okay . Dan: So I get ... Jennifer: So you got ten , looks like sixteen . Dan: Sixteen ? Jennifer: Who you gon na trounce on ? Jennifer: That 's you up there , too , right there , you know . Dan: That 's me right there , too . Jennifer: Oh yeah . Dan: Um ... Jennifer: When w- you take over another person , you take a — you get , their cards . Jennifer: The MSG in that Chinese food really got me high for a little bit . Jennifer: Does MSG affect you ? Dan: No . Dan: Not really . Dan: It affects my mother . Dan: Gives her headaches . Jennifer: Are you gon na attack over there ? Dan: I do n't know . Dan: Thirteen . Dan: That leaves me with thirteen . Dan: I wan na fortify . Jennifer: You ca n't move those to there , because they 're not touching . Dan:  W- w- well that 's kind of bogus . Jennifer: Nun-unh . Dan: Maybe I 'll move em right there . Jennifer: Done . Dan: Done . Jennifer: Oh fuck . Jennifer: Oh . Jennifer: Who 's this guy ? Dan: Player six .\n\n"
        "Example summary: \nTwo people are playing a strategy game online involving cards and attacking countries, while discussing dinner plans.\n\n"
        "Example entities:\nWe\nyou\nI\ncountries you own\ndinner\na percentage of the amount of countries you own\nplayer six\ntheir cards\none of each kind of card\n\n"
        "Answer:\nWe\nyou\nI\ncountries you own\ndinner\ntheir cards\n\n"
        "Note that 'we', 'you' and 'I' are correct answers because the summary mentioned 'two people', which these pronouns refer to in the document.\n\n"
        "Here is the actual document and summmary for entity alignment: \n\n"
        "Document:\n{doc_text} \n\n"
        "Summary:\n{summary} \n\n"
        "Entities:\n{entities} \n\n"
        "Which of the entities in the Entities list also appears in the Summary? Be very precise and only return entities that are mentioned in the summary, separated by new lines. Do not add extra or unrelated entities.\n\nAnswer: "
    )
    if style == 2:
        prompt_template = (
            "Given the following document and summary, which noun phrases are referred to in both the document and the summary? References in the summary do not have to be identical - they can be paraphrases, but must refer to the same entity. For example, if a document mentions 'US President Barack Obama' and the summary mentions 'President Obama', then the noun phrase 'US President Barack Obama' has been referred to in the summary. Similarly if the document says 'Jane Smith: I want to go.' and the summary mentions Jane, then 'Jane Smith' and 'I' have been referred to. Also include nested mentions and abstract phrases - for example if 'austerity measures' are mentioned in a document and summary, then both 'austerity' and 'austerity measures' are mentioned noun phrases. If a noun phrase has articles or modifiers, include them - for example if 'The restaurant' appears in the document and 'a restaurant' appears in the summary, return 'The restaurant', not just 'restaurant'; for 'A number of albums that I like , like Fame', return the whole phrase 'A number of albums that I like , like Fame'.\n\n"
            "Summary:\n\n{summary}\n\n"
            "Document:\n\n{doc_text}\n\n"
            "Task:\n\n"
            "Which noun phrases in the document are mentioned in the summary? Be very precise and exhaustive, going over each phrase in the document and outputting it only if it is referred to in the summary - one phrase from the document per line exactly as it is spelled in the document and nothing else. Make sure that each answer actually appears exactly as printed in the document, and include articles and modifiers if they appear, preserving upper/lower case.\n\n"
            "List of document noun phrases which were referred to in the summary:\n\n"
        )
        use_cached_prompts = False  # No need for cache in this style since this is zero shot

    results = {}
    prompt_cache = defaultdict(dict)
    for line in open("prompts1.tab").read().strip().split("\n"):
        fields = line.split("\t")
        prompt_cache[fields[0]][fields[1]] = fields[2]

    # Lowercase all words in doc_mentions, summary_text, and doc_text for normalization
    doc_mentions_ = [[(span, idx, coref) for span, idx, coref in mentions] for mentions in doc_mentions.values()]
    summary_text_ = []
    for doc in sorted(summary_text):
        summary_text_.append([])
        for source in summary_text[doc]:
            summary_text_[-1].append(summary_text[doc][source])
    doc_text_ = []
    for doc in sorted(doc_text):
        doc_text_.append(doc_text[doc])

    # Process each summary
    docnames = list(doc_mentions.keys())
    for doc_idx, doc_summaries in enumerate(summary_text_):
        docname = docnames[doc_idx]
        sys.stderr.write(f"o LLM: processing {docname} (doc {doc_idx + 1} / {len(docnames)})\n")
        summary_results = []
        for sum_idx, summary in enumerate(doc_summaries):
            if sum_idx == 0:
                pass
                #continue
            if sum_idx > 1:
                pass
                #break
            doc = doc_mentions_[doc_idx]
            total_entities = len(doc)

            # Split doc_mentions into non-overlapping chunks of 15 entities
            entity_indices = list(range(total_entities))
            random.shuffle(entity_indices)  # Shuffle the indices for randomness

            # Create chunks of max chunk_size entities (non-overlapping)
            chunk_size = 15
            chunks = [entity_indices[i:i + chunk_size] for i in range(0, total_entities, chunk_size)]

            if style == 2:  # Just one big chunk
                chunks = [entity_indices]

            all_extracted_mentions = []  # To store the results for each document

            if use_cached_prompts:
                genre = docname.split("_")[1]
                options = prompt_cache[genre]
                for option in options:
                    if option != docname:
                        prompt_template = options[option].replace("\\n", "\n")

            for chunk in chunks:
                # Select the entities corresponding to the current chunk
                selected_entities = [doc[i] for i in chunk]
                entities_str = "\n".join(set([span for span, _, _ in selected_entities]))  # Join entities into a single string with no repetitions

                prompt = prompt_template.format(
                    doc_text=doc_text_[doc_idx],
                    summary=summary,
                    entities=entities_str
                )

                # Making a chat completion request using the client object
                chat = False
                live = False  # Set to True to use the live API
                if live:
                    if chat:
                        response = client.chat.completions.create(
                            model=gpt,  # e.g. use the gpt-4o chat model
                            messages=[
                                {"role": "system", "content": "You are an assistant for aligning entity mentions between documents and summaries."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=300,  # Adjust as necessary to handle multiple mentions
                            temperature=0.2,  # Lower temperature for more deterministic results
                            #top_p=0.7  # Lower top_p for higher precision and less diversity
                        )
                        # Extract and parse the model response
                        answer = response.choices[0].message.content.strip().split("\n")
                    else:
                        response = client.completions.create(
                            model="gpt-3.5-turbo-instruct",
                            prompt=prompt,
                            max_tokens=300,  # Adjust as necessary to handle multiple mentions
                            temperature=0.2,  # Lower temperature for more deterministic results
                            #top_p=0.7  # Lower top_p for higher precision and less diversity
                        )
                        # Extract and parse the model response
                        answer = response.choices[0].text.strip().split("\n")

                else:
                    answer = str(doc_idx) + "-" + str(sum_idx)
                cleaned_ans = [s.lstrip("- ").replace("**", "").replace("*", "").replace('\n', '').strip() for s in answer]

                # Extract mentions from the API response
                extracted_mentions = []
                for ans in cleaned_ans:
                    for span, idx, coref in selected_entities:
                        if ans == span:
                            extracted_mentions.append((span, idx, coref))
                            break
                # Store the extracted mentions
                all_extracted_mentions.extend(extracted_mentions)
            # Append the combined results from multiple queries for one document
            summary_results.append(all_extracted_mentions if all_extracted_mentions else [])

        results[docname] = summary_results

    results = replace_empty_strings(results)  # Replace empty strings with "_"
    return results

def align_llm_hf(doc_mentions, summary_text, model_name="google/flan-t5-xl"):
    """
    Align mentions using a Huggingface model.

    Args:
        doc_mentions (list of list of tuples): List of lists of tuples where each tuple contains (word_span, word_index, coref_index).
        summary_text (list of list of str): List of lists of summaries.
        model_name (str): Name of the Huggingface model to use.

    Returns:
        list of list of list of tuples: A list of lists of lists of tuples where each tuple's `word_span` is found in the corresponding document.
    """
    aligner = pipeline("text2text-generation", model=model_name, device=0) # Run on GPU
    
    prompt_template = (
        "Document: {doc_text}\n"
        "Summary: {summary}\n"
        "For each entity in the document, determine if it aligns with (or makes an equivalent reference to) any word span in the summary. "
        "Return a list of matching word spans from the document."
    )

    results = []

    # Lowercase all words in doc_mentions and summary_text
    doc_mentions_lower = [[(span.lower(), idx, coref) for span, idx, coref in mentions] for mentions in doc_mentions]
    summary_text_lower = [[summary.lower() for summary in summaries] for summaries in summary_text]

    # Extract each summary through all documents to a list of summaries
    num_summaries = len(summary_text_lower[0])
    summaries_by_index = [[] for _ in range(num_summaries)]
    
    for doc_summaries in summary_text_lower:
        for i, summary in enumerate(doc_summaries):
            summaries_by_index[i].append(summary)
    
    # Process each list of summaries
    for summary_idx in range(num_summaries):
        summary_results = []
        for doc_idx in range(len(doc_mentions_lower)):
            summary = summaries_by_index[summary_idx][doc_idx]
            doc = doc_mentions_lower[doc_idx]
            prompt = prompt_template.format(
                doc_text=" ".join([span for span, _, _ in doc]),
                summary=summary
            )

            response = aligner(prompt, max_length=150, num_return_sequences=1)

            answer = response[0]['generated_text'].strip().split("\n")
            extracted_mentions = []

            for ans in answer:
                for span, idx, coref in doc:
                    if ans in span:
                        extracted_mentions.append((span, idx, coref))
                        break

            # Append extracted mentions or empty list
            summary_results.append(extracted_mentions if extracted_mentions else [])
        
        results.append(summary_results)

    return results


def align_stanza(summary_text, doc_text, package="udcoref_xlm-roberta-lora", summary_side="suffix"):
    if len(doc_text) == 0:
        return {}

    import stanza
    if package.endswith(".pt"):
        coref = stanza.Pipeline("en", processors="tokenize,coref", coref_model_path=package)
    else:
        coref = stanza.Pipeline("en", processors="tokenize,coref", package={"coref": package})

    output = {}
    for i, docname in enumerate(doc_text):
        doc = doc_text[docname]
        sys.stderr.write(f"o Stanza: processing {docname}(doc {i + 1} / {len(doc_text)})\n")
        doc_output = []
        for summary_source in summary_text[docname]:
            summary = summary_text[docname][summary_source]
            summary_output = []

            # Tokenize the summary and prepare for coreference
            # Concatenate document and summary text with a marker
            if summary_side == "suffix":
                tokenized_doc_with_summary = doc.strip() + ". === " + summary.strip()
            else:
                tokenized_doc_with_summary = summary.strip() + " === " + doc.strip()
            doc_coref = coref(tokenized_doc_with_summary)

            # Identify tokens that belong to the summary section
            tok_id = 0
            summary_token_start = summary_sent_start = 0
            sent_offset = 0
            for sent in doc_coref.sentences:
                for tok in sent.words:
                    if tok.text == "===":
                        summary_token_start = sent_offset + tok_id + 1
                        summary_sent_start = sent.index
                        break
                    tok_id += 1
                sent_offset += len(sent.words)

            # Extract mentions that have antecedents in the document section
            for coref_chain in doc_coref.coref:
                # Check if any mention exists in both document and summary sections
                doc_mentions_in_chain = []
                summary_mentions_in_chain = []
                for mention in coref_chain.mentions:
                    if summary_side == "suffix":
                        if (mention.sentence >= summary_sent_start and sent_offset + mention.start_word >= summary_token_start) or \
                                mention.sentence > summary_sent_start:
                            summary_mentions_in_chain.append(mention)
                        else:
                            doc_mentions_in_chain.append(mention)
                    else:  # Prefix, summary precedes document body
                        if mention.sentence < summary_sent_start or \
                                (mention.sentence == summary_sent_start and mention.start_word < summary_token_start):
                            summary_mentions_in_chain.append(mention)
                        else:
                            doc_mentions_in_chain.append(mention)

                # Only add document mentions that have corresponding summary mentions
                if summary_mentions_in_chain and doc_mentions_in_chain:
                    # Append mentions from document section to summary_output if they appear in summary section
                    for mention in doc_mentions_in_chain:
                        mention_text = " ".join(
                            doc_coref.sentences[mention.sentence].words[i].text for i in range(mention.start_word, mention.end_word)
                        )
                        indices = ",".join(f"{mention.sentence + 1}-{i + 1}" for i in range(mention.start_word, mention.end_word))
                        summary_output.append((mention_text, indices, str(coref_chain.index)))

            # Append only if summary_output has content to ensure unique doc_output
            doc_output.append(summary_output)
        output[docname] = doc_output

    return output


def align(doc_first_mentions, summary_text, doc_all_mentions, doc_text, component="string_match"):
    if component == "LLM":
        return align_llm(doc_first_mentions, summary_text, doc_text)
    elif component == "LLM_zero":
        return align_llm(doc_all_mentions, summary_text, doc_text, style=2)
    elif component == "LLM_hf":
        return align_llm_hf(doc_first_mentions, summary_text)
    elif component == "stanza":
        return align_stanza(summary_text, doc_text)
    elif component == "stanza_pre":
        return align_stanza(summary_text, doc_text, summary_side="prefix")
    elif component == "stanza_on":
        return align_stanza(summary_text, doc_text, package="ontonotes-singletons_roberta-large-lora")
    elif component == "stanza_onpre":
        return align_stanza(summary_text, doc_text, package="ontonotes-singletons_roberta-large-lora", summary_side="prefix")
    elif component == "stanza_gum":
        return align_stanza(summary_text, doc_text, package="gum_roberta-large-lora")
    elif component == "stanza_gumpre":
        return align_stanza(summary_text, doc_text, package="gum_roberta-large-lora", summary_side="prefix")
    else:
        raise ValueError(f"Unknown alignment component: {component}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align document mentions based on the selected component")
    parser.add_argument("--model_name", default="gpt4o", choices=["gpt4o", "claude-3-5-sonnet-20241022","meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"], help="Model name to use for summarization")
    parser.add_argument("--max_docs", type=int, default=2, help="Maximum number of documents to processe (default: None = all; choose a small number to prototype)")
    parser.add_argument("--component", required=False, default="string_simple", choices=["LLM", "stanza", "stanza_on","stanza_pre","string_simple"], help="Component to use for alignment")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached summaries (default: False)")
    parser.add_argument("--summary_cache", action="store_true", help="Use cached summaries (default: False)")
    parser.add_argument("--doclist", default=None, help="Optional file with document names to process, one per line")

    args = parser.parse_args()

    doc_ids = None
    if args.doclist is not None:
        with open(args.doclist, 'r') as file:
            doc_ids = [line.strip() for line in file]

    doc_ids, doc_texts = read_documents(doc_ids)

    all_entities_from_tsv = get_entities_from_gold_tsv(gum_src + "tsv", docnames=doc_ids)
    gold_summaries = extract_gold_summaries_from_xml(gum_src + "xml", docnames=doc_ids)

    if args.max_docs is not None:
        doc_ids = doc_ids[:args.max_docs]
        doc_texts = doc_texts[:args.max_docs]
        print('doc_ids:',doc_ids)
    else:
        args.max_docs = len(doc_ids)

    if args.summary_cache:
        summaries = defaultdict(dict)
        lines = open("summaries_final.txt", "r").read().split("\n")
        docname = ""
        for line in lines:
            if line[:4] in ["GUM_", "GENT"]:
                docname = line.strip()
                human_idx = 1
            elif line.strip() != "":
                if docname not in doc_ids:
                    continue
                model, summary = line.split(")", 1)
                model = model.replace("(", "").strip()  # Clean brackets
                if "human" in model:
                    model = "human" + str(human_idx)
                    human_idx += 1
                summaries[docname][model] = summary.strip()
    else:
        if args.model_name == "gpt4o":
            summaries = get_summary_gpt4o(doc_texts, doc_ids, ".", partition="all", model_name=args.model_name, n=1, overwrite=args.overwrite_cache)
        elif args.model_name == "claude-3-5-sonnet-20241022":
            summaries = get_summary_claude35(doc_texts, doc_ids, ".", partition="all", model_name=args.model_name, n=1, overwrite=args.overwrite_cache)
        else:
            summaries = get_summary(doc_texts, doc_ids, ".", partition="all", model_name=args.model_name, n=1, overwrite=args.overwrite_cache)

    all_mentions_from_tsv = get_entities_from_gold_tsv(gum_src + "tsv", docnames=doc_ids, first=False)
    all_mentions_from_tsv = {doc_ids[i]: all_mentions_from_tsv[i] for i in range(len(doc_ids))}

    # Get all mentions from each document
    all_mentions = defaultdict(set)
    for docname in doc_ids:
        for mention in all_mentions_from_tsv[docname]:
            all_mentions[docname].add(mention)

    doc_sp_texts = extract_text_speaker_from_xml(gum_src + "xml", docnames=doc_ids)[:args.max_docs]
    doc_sp_texts = {doc_ids[i]: doc_sp_texts[i] for i in range(len(doc_sp_texts))}

    alignments = align(
        doc_first_mentions=all_entities_from_tsv[:args.max_docs],
        summary_text=summaries,
        doc_all_mentions=all_mentions,
        doc_text=doc_sp_texts,
        component=args.component,
    )

    print(f"{args.component}:\n{alignments}")
