import os, re, sys
from glob import glob
from random import choice
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoModelForCausalLM
import pandas as pd
import xml.etree.ElementTree as ET
from openai import OpenAI
import anthropic
from huggingface_hub import InferenceClient
from apis import gpt4_key, claude_key, hf_token

client = OpenAI(api_key=gpt4_key)
claude_client = anthropic.Anthropic(api_key=claude_key)
hf_client = InferenceClient(token=hf_token,timeout=120.0)

examples = {'academic': {'GUM_academic_exposure': 'This study shows that limited exposure to a second language (L2) after it is no longer being actively used generally causes attrition of L2 competence.', 'GUM_academic_librarians': 'This poster paper presents a plan to hold up to six Digital Humanities (DH) clinics, one day events which include lectures and hands-on training, to help Dutch librarians in the Netherlands and Belgium to provide researchers and students with services and follow literature and research in the area of DH.'}, 'bio': {'GUM_bio_byron': "Some details of Lord Byron's early life including his education at Aberdeen Grammar School, Harrow and Trinity College, as well as his romantic involvements and friendships with Mary Chaworth, John FitzGibbon and John Edleston.", 'GUM_bio_emperor': 'Joshua Norton was an eccentric resident of San Francisco who proclaimed himself Emperor of the United States in the second half of the 19th Century, and came to be known as Emperor Norton.'}, 'conversation': {'GUM_conversation_grounded': "After being grounded for staying out at night, Sabrina fights with her mother, who does not permit her to join the volleyball team and later talks to her partner about cleaning the house's insulation to prevent mice from spreading an ongoing Hantavirus epidemic.", 'GUM_conversation_risk': 'Two people are playing a strategy game online involving cards and attacking countries, while discussing dinner plans.'}, 'court': {'GUM_court_loan': 'General Elizabeth B. Prelogar, representing the Biden administration before the Chief Justice and the Court in a challenge by Nebraska and other states, argues that the HEROES Act authorizes Secretary Cardona to suspend payment obligations for federal student loans since it expressly allows waiver or modification of any Title IV provision in a national emergency, like COVID-19.', 'GUM_court_negligence': "In an appeal by Atlantic Lottery Corporation against Douglas Babstock et al., attorney Julie Rosenthal argues that the Newfoundland and Labrador Court of Appeal's ruling that negligence can amount to wrongdoing in the absence of loss or injury is new and meets no unmet societal need, while contradicting precedent from the law of negligence, and should therefore be overturned."}, 'essay': {'GUM_essay_evolved': 'Psychiatrist Arash Javanbakht suggests that to feel happier, humans must live the life they evolved to live, which includes physical activity, avoiding high-calorie sugary foods that were not available to our ancestors, changing our behavior around sleep by avoiding late caffeine and screens, and exposing ourselves to a healthy dose of excitement and a bit of fear.', 'GUM_essay_tools': 'An anonymous writer presents the example of the changes to the automotive industry, in particular the trend toward computerized cars, to argue that technology, rather than being a universal good that empowers the masses, can instead be and has been utilized to give large corporations and governments greater and greater control at the expense of individual consumers and small businesses; therefore we must consider &amp;quot;who is devising the technology and why,&amp;quot; and push for open systems and decentralized technologies.  '}, 'fiction': {'GUM_fiction_beast': 'A thirteen year old girl goes to mass with her father to take communion on a rainy day in March, then speculates who or what might be making two frightening noises she hears on her way back home as they pass through her school gym to get out of the rain.', 'GUM_fiction_lunre': 'A protagonist recounts the day when his father, a patriotic islander, returned from a long journey and unexpectedly brought home a foreign Olondrian tutor from Bain named Master Lunre.'}, 'interview': {'GUM_interview_cyclone': "Wikinews interviews several meteorologists about the prognosis for Cyclone Phailin as it approaches land in the Bay of Bengal at 190 km/h, and the Indian government's preparedness for the storm given past deaths caused by cyclones in the area.", 'GUM_interview_gaming': 'In an interview with Wikinews, Mario J. Lucero and Isabel Ruiz, who run a website called Heaven Sent Gaming, talk about how they met in school, the people involved in their project and the motivation for their work.'}, 'letter': {'GUM_letter_arendt': "On August 19, 1975, Bill writes a letter to Hannah, where he describes his stay at his friend's villa in Mallorca, details a conversation about war and politics with a Spanish prince at the villa, compares financial life in New York to London, updates her on his illness, tells her the books he is reading and editing, and asks how she is and how her manuscript has progressed.", 'GUM_letter_wiki': 'In a May 2006 letter to Penn State Undergraduate Dean Robert Pangborn, student and wiki advocate George Chriss defends his proposal for a student-based university wiki as a complement to the University’s official course description and effective course feedback pipeline, after Penn’s decision against wiki on grounds of potential misinformation and management difficulties.'}, 'news': {'GUM_news_homeopathic': 'Indian Australian couple Thomas and Manju Sam are being prosecuted in Australia for manslaughter by gross negligence in the death of their nine-month old daughter Gloria, whose severe eczema they refused to treat using conventional medicing, instead opting for homeopathic treatments common in India, which are known to be ineffective.', 'GUM_news_iodine': 'A 2006 Australian study has shown that almost half of all Australian children suffer from mild iodine deficiency, which can cause health problems especially for children and pregnant women, and is probably due to the replacement of iodine-containing sanitisers in the dairy industry and the lack of iodized salt in Australia.'}, 'podcast': {'GUM_podcast_bangladesh': 'In this podcast, the editors of the Global Voices podcast discuss with Rezwan and Pantha about Mangal Shobhajatra, a festive annual procession with singing, dancing, and colorful displays for the Bengali New Year that takes place in Dhaka, Bangladesh and serves as a symbol of unity and diversity in Bangladesh.', 'GUM_podcast_wrestling': "On the show Beyond the Mat, Dave and Alex discuss wrestling news in the week of WrestleMania, Raw and Smackdown, including being glad that their favorite wrestler the Undertaker has finally retired despite people's objections and 90s nostalgia, and Alex recounts drinking tequila on his birthday and discusses the new DLC for King of Fighters XIV, which includes Rock Howard."}, 'reddit': {'GUM_reddit_macroeconomics': 'In a post answering the question how and to whom countries can be in debt, the author explains how money is a form of debt whose value depends on the amount of money in circulation, and how fiat currency developed as a substitute for gold reserves which formerly backed bank debt certificates.', 'GUM_reddit_pandas': 'Some Reddit forum users discuss whether humans are the only species which practices birth control to prevent reproduction, leading to a discussion of whether or not pandas are poor at reproducing, and some other animals which may become less reproductive when food is scarce, such as rats and rabbits.'}, 'speech': {'GUM_speech_impeachment': 'In a speech in the US Congress, a Democratic member of Congress accuses Republican Senators of failing to fulfill their oath to conduct an impartial trial in the impeachment of President Donald Trump for abuse of power and attempts to solicit foreign interference in the 2020 elections.', 'GUM_speech_inauguration': 'In his inaugural address, US President Ronald Reagan praises the peaceful transition of power to a new Presidency and lays out his plans to reduce the role of government and spending in order to revive the economy and combat inflation and unemployment.'}, 'textbook': {'GUM_textbook_governments': "This section of a textbook explains and exemplifies different types of government, including representative democracy as in the United States, direct democracy as in ancient Athens, monarchy as in Saudi Arabia, oligarchy as in Cuba's reigning Communist Party, and totalitarianism as in North Korea.", 'GUM_textbook_labor': 'This excerpt explains three reasons why specialization of labor increases production: it allows workers to specialize in work they have a talent for, it allows them to improve particularly in certain tasks, and it allows businesses to take advantage of economies of scale, such as by setting up assembly lines to lower production costs.'}, 'vlog': {'GUM_vlog_portland': "In a video, Katie tells about her vacation to Portland, Oregon, and gives her top 4 recommendations for the region: Crater Lake, shopping at Powell's indie bookstore which sells used and new books, hiking in Forest Park and visiting the Rose Garden and the Japanese Gardens.", 'GUM_vlog_radiology': 'A radiology resident vlogging on YouTube tells about his week on general nuclear medicine rotation, where he did three lymphoscintigraphies and some ultrasounds, his plans to work out after he gets off early from work, and taking Dutch cough drops to treat his sore throat ahead of a big trip he is planning the following weekend.'}, 'voyage': {'GUM_voyage_athens': 'This article presents Athens, an ancient city and capital of modern Greece with a metropolitan population of 3.7 million, which hosted the 2004 Olympic games and features archeological sites and restored neoclassical as well as post-modern buildings, and is best visited in spring and late autumn.', 'GUM_voyage_coron': 'This overview of the history of Coron, a fishing town on the island of Busuanga in the Philippines, tells about the local people (Tagbanuas), the shift from farming to mining, fishing, and more recently tourism, and attractions such as stunning lakes, snorkeling, and wreck diving to see around ten Japanese ships sunk by the US Navy in World War II.'}, 'whow': {'GUM_whow_joke': "This section from a guide on how to tell a joke advises practicing but not memorizing jokes, knowing your audience to pick appropriate jokes, choosing material from your life, online or repurposing jokes you've heard, using a realistic but exaggerated setup your audience can relate to, followed by a surprising punchline and possibly an additional 'topper' punchline.", 'GUM_whow_overalls': "This guide to washing overalls suggests washing them with like clothing, avoiding clothes which can get twisted up with the straps, fastening straps to the bib with twist ties (also in the dryer), emptying pockets, moving the strap adjusters to make them last longer, using less detergent if washing overalls alone, and taking care plastic ties don't melt in the dryer."}, 'dictionary': {'GENTLE_dictionary_next': "This dictionary entry for the word 'next' gives its dialectal forms, etymology as a superlative of 'near' from Middle and Old English nexte and nīehsta from Proto-Germanic *nēhwist, its cognates (e.g. Dutch naast, Danish næste), pronunciation /nεkst/ and related synonyms and antonyms, accompanied by usage examples as an adjective, determiner, adverb, preposition and noun.", 'GENTLE_dictionary_school': "This dictionary entry for the word 'school' gives its pronunciation /skuːl/, etymologies for its senses a group/to form a group of fish (from Middle English scole, Proto-Germanic *skulō 'crowd') and elementary school/to educate (from Proto-Germanic *skōla, from Late Latin schola, Ancient Greek σχολεῖον), along with cognates and usage examples as both noun and verb.", 'GENTLE_dictionary_trust': "This dictionary entry for the word 'trust' provides its etymology from Middle English 'trust' and Proto-Indo-European *deru- 'be firm', cognates such as Danish trøst, pronunciations like /trʌst/ and /trʊst/ and synonyms, antonyms, usage examples and definitions as a noun (e.g. 'confidence', 'hope'), senses in computing and law, and as a verb meaning to have faith or believe."}, 'esports': {'GENTLE_esports_fifa': "This episode features three games towards winning the Scudetto, where the player wins a simulated game against Benevento 3:1, then plays and loses a game to Genoa 0:1 due to a penalty goal from Favilli, but wins a simulation against Manchester 3:1 with Paquetá and Jović scoring, making the semi-finals and fulfilling the club's objective for the Champions League this season.", 'GENTLE_esports_fortnite': 'In the Fortnite World Cup finals with 100 players on screen, Jack and another person discuss 16 year old Sentinels player Bugha, the current world champ who may win $3 million, while Skite is defeated, Maufin and Zayt fight by Loot Lake, BlastR fights outside Pressure Plant, letw1k3 beats JarkoS, and Pika, ranked 11th, takes down a player in Megamall, then tracks Kinstaar.'}, 'legal': {'GENTLE_legal_abortion': "In Roe v. Wade, an appeal before the US Supreme Court (argued 1971, decided 1973) the Court ruled in favor of Roe, a pregnant woman who, with other plaintives, had brought a class action suit against a Texas law proscribing abortion except to save the mother's life, declaring the statute unconstitutional as infringing on the 9th and 14th amendments.", 'GENTLE_legal_service': 'This services agreement between StartEngine Crowdfunding, Inc. and Solutions Vending Internaitonal, Inc. CEOs Dawn Dickson and Howard Marks on 08/19/2019 specifies the conditions for services to present its securities offering to users and manage user accounts and subscriptions, including fees and billing, reimbursable expenses, warranties and terms to terminate the agreement.'}, 'medical': {'GENTLE_medical_anemia': 'After examining a 72 year old man (allergic to Vicodin) with chronic lymphocytic leukemia (CLL) since May 2008 and currently taking 5mg predinsone every other day, a physician reports he is suffering from autoimmune hemolytic anemia and oral ulcers on the lip and tongue, and puts him back on Valtrex and lowers steroids to 1mg daily ahead of a clinic visit the next week.', 'GENTLE_medical_hiv': 'A physician report on examining a 41-year-old white male with HIV disease (stable control on Atripla), resolving left gluteal abscess, diabetes on oral therapy, hypertension, depression and chronic pain, and plans to continue medication (Gabapentin, Metformin, Glipizide, Atripla etc.) and psychological treatment, order lab work in 3-4 weeks and then see him a few weeks later.', 'GENTLE_medical_screw': "A 59-year-old man's followup 4 months post percutaneous screw fixation of a right Schatzker IV tibial plateau fracture and 2nd-5th metatarsal head fractures shows good pain control and incision and fracture healing, but decreased foot sensation (L4-L5 distribution bilaterally) and bridging callus, recommending physical therapy and neurologist/surgeon if paresthesias worsen.", 'GENTLE_medical_transplant': 'This report on a followup and chemotherapy visit for 51-year-old white male patient with posttransplant lymphoproliferative disorder (diagonosed 2007) documents chronic renal insufficiency, squamous cell carcinoma of the skin, anemia and hypertension, as well as diffuse large B-cell lymphoma following transplantaiton and suggests followup in 3 weeks.'}, 'poetry': {'GENTLE_poetry_annabel': 'This poem tells of the strong childhood love of the narrator and the beautiful Annabel Lee many years ago in a kingdom by the sea, which led the envious angels of heaven to chill and kill her via a wind out of a cloud at night, though their souls cannot be dissevered and the narrator continues to dream of her and feel her bright eyes.', 'GENTLE_poetry_death': "On a slow carriage ride with Death and Immortality, the narrator passes by the school, the fields of grain and the setting sun, then pauses before a house that seemed a swelling of the ground, where centuries later it feels shorter than the day the narrator realized the horses' heads were toward eternity. ", 'GENTLE_poetry_flower': "In this poem, the narrator tells of hiding inside their flower, which the poem's addressee, unsuspecting, wears, while feeling almost a loneliness for the narrator.", 'GENTLE_poetry_raven': "After reading and napping on a bleak December night to forget the sorrow of losing Lenore, a rare and radiant maiden, a raven comes in through the window and perches above the chamber door, repeatedly saying 'Nevermore' when asked for its name and about whether the narrator will again clasp Lenore, upsetting the narrator greatly.", 'GENTLE_poetry_road': 'In this poem, a traveller recounts deliberating which of two roads diverging in a yellow wood to take, finally choosing the one less travelled by, because it was grassy and wanted wear, and predicts one day telling somewhere that this choice had made all the difference.'}, 'proof': {'GENTLE_proof_five': 'This mathematical proof shows by induction that for any graph Gr of n vertices a proper vertex k-coloring can be assigned such that k≤5 colors, using the Minimum Degree Bound for Simple Planar Graph theorem, which states that Gr+1 has at least one vertex x with at most 5 edges, whose removal would yield a 5-colorable graph, allowing one of 5 colors for x in each condition.', 'GENTLE_proof_square': 'This proof demonstrates that if AB is a second bimedial line divided at C into its medials, and DE is a rational line where DEFG = AB² is applied to DE such that DG is its breadth, then DG is a third binomial straight line.', 'GENTLE_proof_wosets': 'This mathematical proof shows that two wosets S and T, ordered sets with a well-ordering, must be order isomorphic to each other, or else one set is order isomorphic to an initial segment of the other, trivially if they contain less than two elements, and otherwise provably using a relation ⪯, which is shown to be a reflexive, transitive and antisymmetric total well-ordering.'}, 'syllabus': {'GENTLE_syllabus_opensource': 'This syllabus outlines CSCI-4470-01, an Open Source Software course taught in Spring 2022 at RPI by Prof. Wesley Turner and two TAs on Tuesdays and Fridays, which teaches students about open source, new technology, Good Code, collaborating, patching open source projects, diversity issues and RCOS project preparation, and covers Licensing, Version Control and common tools.', 'GENTLE_syllabus_techtonica': 'This curriculum for Techtonica, a non-profit project offering hands on tech training, details a 15 week syllabus covering developer tools, webpages (HTML, CSS), JavaScript at React, Web APIs, databases, testing, data structures and algorithms, an Eventonica project and full stack practice, as well as a career week and assessments.'}, 'threat': {'GENTLE_threat_bolin': "In a letter to Jerry on 6/25/90, the author threatens Jerry for hitting their daughter Paula and underreporting how much he got for some vehicles, and then demands that Jerry return the author and Paula's things such as clothes, books, a gun, TV and VCR to Paula within a week or suffer the consequences.", 'GENTLE_threat_dillard': 'In this threat letter, Angel Dillard admonishes Dr. Means for carrying out abortions at the Harry St. clinic in Wichita, which are portrayed as killing children, arguing that other physicians will avoid associating with Dr. Means, and invoking the threat of a car bomb, picketing at home and at the office, loss of clientele and staff, as well as religious objections to abortion.', 'GENTLE_threat_kelly': 'In this letter, an inmate in jail threatens a woman, who works at Walgreens and has perpetrated an insurance scam and stolen some things, that they will reveal this to the police and testify against her unless the woman drops the charges that put the author of the letter in jail, resulting in 3-5 years in jail for a class E felony.', 'GENTLE_threat_malik': 'In this letter, the author, who is in prison, pleads for a reversal of their court case while accusing the two White Jewish judges responsible of being racially motivated, and threatening them with armed robbery of $20,000 in cash, as well as intending to submit 5 lawsuits in the Northern District Court in the next two weeks.', 'GENTLE_threat_white': 'In this letter, the writer tells the addressee, who is in the collections business, that they have hired someone to find them, and threatens to disclose damaging information to their Citibank customers, as well as publicizing information on outstanding disputed credit accounts, if they do not fax over a certain letter quickly.'}}

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
gum_src = script_dir + ".." + os.sep + ".." + os.sep + "src" + os.sep


def get_example(genre, docname):
    # Get a summary from the same genre but avoiding this exact docname
    global examples
    example_dict = examples[genre]
    other_doc_examples = [v for k, v in example_dict.items() if k != docname]
    return choice(other_doc_examples)


def generate_valid_summary(prompt, model_name, docname, summary_client=None, tokenizer=None):
    attempts = 0
    prefix = ""
    while True and attempts < 5:
        attempts += 1
        if attempts > 1:
            new_words = 50
            if attempts > 2:
                new_words = 45
            prefix = f"The last summary was too long. Please aim for about {new_words} words. "
            prompt = prompt.replace("55 words", f"{new_words} words")
        if "gpt" in model_name:
            response = summary_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a summarization assistant generating concise one-sentence summaries."},
                    {"role": "user", "content": prefix + prompt}
                ]
            )
            summary = response.choices[0].message.content.strip()
        elif "claude" in model_name:
            response = summary_client.messages.create(
                model=model_name,
                max_tokens=120,  # Adjust as needed
                system="You are a summarization assistant generating concise one-sentence summaries.",
                messages=[
                    {"role": "user", "content": prefix + prompt},
                    {"role": "assistant", "content": "Here is the summary:"}
                ]
            )
            summary = ''.join([block.text for block in response.content])

        elif "flan" in model_name:  # Local Huggingface model
            out = client.generate(
                **input_ids, max_new_tokens=120, num_return_sequences=1, do_sample=True,
                eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1
            )
            summary = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        else:
            response = hf_client.chat_completion(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a summarization assistant generating concise one-sentence summaries."},
                    {"role": "user", "content": prefix + prompt}
                ]
            )
            summary = response.choices[0].message.content.strip()

        if "\n\n" not in summary and 50 <= len(summary) <= 380:
            return summary.strip()
        print(f"Reprompting {model_name}: Invalid formatting or invalid length {len(summary)} for document {docname}")
    print(f"Failed to generate a valid summary for document {docname} in {attempts} attempts.")
    quit()


def get_summary(doc_texts, doc_ids, data_folder, partition, model_name="google/flan-t5-xl", n=1, overwrite=False):
    global examples

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8"
    )

    model_name_short = model_name.split("/")[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "flan" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

    if not data_folder.endswith(os.sep):
        data_folder += os.sep
    summary_folder = data_folder + "output" + os.sep + "summaries" + os.sep + partition + os.sep + model_name_short + os.sep
    os.makedirs(summary_folder, exist_ok=True)

    all_summaries = {}
    cached_summaries = 0
    cached_summary_docs = 0
    written_summaries = 0
    written_summary_docs = 0

    for i, doc_text in enumerate(doc_texts):
        doc_id = doc_ids[i]
        doc_summaries = []

        summaries_exist = all(
            os.path.exists(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt") for j in range(n)
        )
        
        if summaries_exist and not overwrite:
            for j in range(n):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt", "r", encoding="utf-8") as f:
                    doc_summaries.append(f.read().strip())
                    cached_summaries += 1
            cached_summary_docs += 1
        else:
            genre = doc_id.split("_")[1]
            example = get_example(genre, doc_id)
            prompt = f"Summarize the following document in 1 sentence. Make sure your summary is one sentence long and does not exceed 380 characters. Example of summary style: {example}\n\nDocument:\n\n{doc_text}\n\nPlease output just the summary and nothing else. Summary:"

            input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

            doc_summaries = [generate_valid_summary(prompt, model_name, doc_id, summary_client=model, tokenizer=tokenizer) for _ in range(n)]

            for k, summary in enumerate(doc_summaries):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{k}.txt", "w", encoding="utf-8", newline="\n") as f:
                    f.write(summary.strip())
                    written_summaries += 1
                    print("Wrote summary for", doc_id)
            written_summary_docs += 1
        
        all_summaries[doc_id] = doc_summaries

    if cached_summaries > 0:
        sys.stderr.write(f"Loaded {cached_summaries} cached summaries for {cached_summary_docs} documents.\n")
    if written_summaries > 0:
        sys.stderr.write(f"Wrote {written_summaries} new summaries for {written_summary_docs} documents.\n")

    return all_summaries


def get_summary_gpt4o(doc_texts, doc_ids, data_folder, partition, model_name="gpt4o", n=4, overwrite=False):
    # Extract the actual model name (strip the company/organization prefix)
    model_name_short = model_name.split("/")[-1]

    if not data_folder.endswith(os.sep):
        data_folder += os.sep
    summary_folder = data_folder + "output" + os.sep + "summaries" + os.sep + partition + os.sep + model_name_short + os.sep

    # Ensure the output directory exists
    os.makedirs(summary_folder, exist_ok=True)

    all_summaries = {}
    cached_summaries = 0
    cached_summary_docs = 0
    written_summaries = 0
    written_summary_docs = 0

    for i, doc_text in enumerate(doc_texts):
        doc_id = doc_ids[i]
        doc_summaries = []

        summary_folder = data_folder + "output" + os.sep + "summaries" + os.sep + partition + os.sep + model_name_short + os.sep

        # Check if summaries exist and load them if overwrite is False
        summaries_exist = all(
            os.path.exists(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt") for j in range(n)
        )
        slug = model_name_short
        if not summaries_exist and "qwen" in model_name_short.lower():
            slug = "Qwen2.5-3B-Instruct"  # See if we have a version from a model with different parameter count
            summaries_exist = all(
                os.path.exists(f"{summary_folder.replace('7B','3B')}{slug}_{doc_id}{j}.txt") for j in range(n)
            )
            summary_folder = data_folder + "output" + os.sep + "summaries" + os.sep + partition + os.sep + slug + os.sep

        if summaries_exist and not overwrite:
            # Load existing summaries
            for j in range(n):
                with open(f"{summary_folder}{slug}_{doc_id}{j}.txt", "r", encoding="utf-8") as f:
                    doc_summaries.append(f.read().strip())
                    cached_summaries += 1
            cached_summary_docs += 1
        else:
            raise ValueError("Missing cached summaries")
            # Generate new summaries
            genre = doc_id.split("_")[1]
            example = get_example(genre, doc_id)
            prompt = f"Summarize the following document in 1 sentence. Make sure your summary is one sentence long and does not exceed 380 characters. Example of summary style: {example}\n\nDocument:\n\n{doc_text}\n\nPlease output just the summary and nothing else. Summary:"

            doc_summaries = [generate_valid_summary(prompt, model_name, doc_id, client) for _ in range(n)]

            # Write summaries in the specified filename format
            for k, summary in enumerate(doc_summaries):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{k}.txt", "w", encoding="utf-8", newline="\n") as f:
                    f.write(summary.strip())
                    written_summaries += 1
            written_summary_docs += 1
        
        all_summaries[doc_id] = doc_summaries

    if cached_summaries > 0:
        sys.stderr.write(f"Loaded {cached_summaries} cached summaries for {cached_summary_docs} documents.\n")
    if written_summaries > 0:
        sys.stderr.write(f"Wrote {written_summaries} new summaries for {written_summary_docs} documents.\n")

    return all_summaries

def get_summary_claude35(doc_texts, doc_ids, data_folder, partition, model_name="claude-3-5-sonnet-20241022", n=4, overwrite=False):

    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "
    # Extract the actual model name
    model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name

    # Ensure the output directory exists
    if not data_folder.endswith(os.sep):
        data_folder += os.sep
    summary_folder = data_folder + "output" + os.sep + "summaries" + os.sep + partition + os.sep + model_name_short + os.sep
    os.makedirs(summary_folder, exist_ok=True)

    all_summaries = {}
    cached_summaries = 0
    cached_summary_docs = 0
    written_summaries = 0
    written_summary_docs = 0

    for i, doc_text in enumerate(doc_texts):
        doc_id = doc_ids[i]
        doc_summaries = []

        # Check if summaries exist in the specified filename format and load them if overwrite is False
        summaries_exist = all(
            os.path.exists(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt") for j in range(n)
        )
        
        if summaries_exist and not overwrite:
            # Load existing summaries
            for j in range(n):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{j}.txt", "r", encoding="utf-8") as f:
                    doc_summaries.append(f.read().strip())
                    cached_summaries += 1
            cached_summary_docs += 1
        else:
            # Generate prompt for the document
            genre = doc_id.split("_")[1]
            example = get_example(genre, doc_id)
            prompt = f"Summarize the following document in 1 sentence. Make sure your summary is one sentence long and does not exceed 380 characters. Example of summary style: {example}\n\nDocument:\n\n{doc_text}\n\nPlease output just the summary and nothing else. Summary:"

            # Call the Claude API to generate multiple summaries if needed
            doc_summaries = [generate_valid_summary(prompt, model_name, doc_id, claude_client) for _ in range(n)]

            # Write summaries in the specified filename format
            for k, summary in enumerate(doc_summaries):
                with open(f"{summary_folder}{model_name_short}_{doc_id}{k}.txt", "w", encoding="utf-8", newline="\n") as f:
                    f.write(summary.strip())
                    written_summaries += 1
            written_summary_docs += 1

        all_summaries[doc_id] = doc_summaries

    if cached_summaries > 0:
        sys.stderr.write(f"Loaded {cached_summaries} cached summaries for {cached_summary_docs} documents.\n")
    if written_summaries > 0:
        sys.stderr.write(f"Wrote {written_summaries} new summaries for {written_summary_docs} documents.\n")

    return all_summaries


def extract_gold_summaries_from_xml(directory, docnames=None):
    # List to store extracted data
    data = []
    
    # Iterate over all XML files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            docname = os.path.basename(filename).replace(".xml", "")
            if docnames and docname not in docnames:
                continue
            filepath = os.path.join(directory, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Extract doc_id (from the 'id' attribute of the root element)
            doc_id = root.get('id')
            
            # Extract the summary text
            summary = root.get('summary', '')
            
            # Append the data to the list
            data.append({'doc_id': doc_id, 'summary': summary})
            
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    result_dict = df.groupby('doc_id')['summary'].apply(list).to_dict()
    
    return result_dict


def extract_text_speaker_from_xml(directory, docnames=None):
    def uncamel(text):
        # Insert spaces into a camel-case string. Single cap letters get a period, e.g.:
        # HaroldMSmith -> Harold M. Smith
        text = re.sub(r'([A-Z])([A-Z])', r'\1.\2', text)
        output = []
        for c in text:
            if c.isupper() and output and output[-1] != " ":
                output.append(" ")
            output.append(c)
        return "".join(output)

    # List to store the concatenated text from each document
    all_documents = []
    
    # Get and sort all XML files in the directory
    xml_files = sorted([f for f in os.listdir(directory) if f.endswith('.xml')])

    # Iterate through sorted XML files
    for filename in xml_files:
        filepath = os.path.join(directory, filename)
        docname = os.path.basename(filename).replace(".xml", "")
        if docnames and docname not in docnames:
            continue

        xml = open(filepath, "r", encoding="utf-8").read()
        speaker_count = int(re.search(r' speakerCount="(\d+)"', xml).group(1))
        output = []
        speaker = prev_speaker = ""
        for line in xml.split("\n"):
            line = line.strip()
            if line.startswith("<sp "):
                speaker = re.search(r' who="#([^"]+)"', line).group(1)
                speaker = uncamel(speaker)
            if speaker_count > 0 and "_fiction_" not in docname:
                if speaker != prev_speaker:
                    output.append(f"{speaker}:")
                    prev_speaker = speaker
            if not (line.startswith("<") and line.endswith(">")) and "\t" in line:
                output.append(line.split("\t")[0])

        # Add the full text of the current document to the list
        all_documents.append(" ".join(output))
    
    return all_documents


def read_documents(doclist=None):
    files = sorted(glob(gum_src + "tsv" + os.sep + '*.tsv'))
    doc_ids = []
    doc_texts = []
    for file_ in files:
        if doclist:
            if os.path.basename(file_).split(".")[0] not in doclist:
                continue
        docname = os.path.basename(file_).split(".")[0]
        sents = re.findall(r'#Text=([^\n]+)', open(file_, "r", encoding="utf-8").read())
        text = " ".join([s.strip() for s in sents])
        doc_texts.append(text)
        doc_ids.append(docname)

    return doc_ids, doc_texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate summaries for GUM documents")
    parser.add_argument("--data_folder", default="data", help="Path to data folder")
    parser.add_argument("--model_name", default="google/flan-t5-xl", choices=["gpt4o", "claude-3-5-sonnet-20241022", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"], help="Model name to use for summarization")
    parser.add_argument("--n_summaries", type=int, default=4, help="Number of summaries to generate per document")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached summaries (default: False)")
    parser.add_argument("--partition", default="train", choices=["test", "dev", "train"], help="Data partition to use for generating and storing summaries")

    args = parser.parse_args()

    doc_ids, doc_texts = read_documents()
    docs = list(zip(doc_texts, doc_ids))

    # Sample just a few docs to test - comment this out to use all documents
    # seed(42)
    # shuffle(docs)
    # doc_texts, doc_ids = zip(*docs)
    # doc_texts = doc_texts[:12]
    if args.model_name=="gpt4o":
        summaries =get_summary_gpt4o(doc_texts, doc_ids, args.data_folder, args.partition, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    elif args.model_name=="claude-3-5-sonnet-20241022":
        summaries =get_summary_claude35(doc_texts, doc_ids, args.data_folder, args.partition, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)
    else:
        summaries = get_summary(doc_texts, doc_ids, args.data_folder, args.partition, model_name=args.model_name, n=args.n_summaries, overwrite=args.overwrite_cache)

    for doc_id, doc_summaries in summaries.items():
        print(f"Document ID: {doc_id}\n")
        for i, summary in enumerate(doc_summaries, 1):
            print(f"Summary {i}: {summary}")
            print()
