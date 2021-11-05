# This script takes a directory of amalgum xml files and for each file it does the following: The xml tag information is extracted
# into dictionaries that are indexed by the sentence indexes and token indexes to which the tags correspond. The xml annotations are 
# then added to the data from the corresponding conllu file, and the result is written to a new file. Finally, a new xml file is 
# reconstructed from the xml annotations in the new conllu file, and evaluated to be the same as the original xml file.

import os
import re
import shlex
from conllu import parse
import html

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
REDUC_DIR = SCRIPT_DIR + 'reduced_amalgum_xml' + os.sep
UPD_DIR = SCRIPT_DIR + 'updated_amalgum_conllu' + os.sep
RECON_DIR = SCRIPT_DIR + 'reconstructed_amalgum_xml' + os.sep
ORIG_DIR = SCRIPT_DIR + 'original_amalgum_conllu' + os.sep

#REDUC_DIR = SCRIPT_DIR + 'test' + os.sep 

# Takes in the file path of an amalgum xml file and returns the following 4 dictionaries:
# sentence_level_info: contains info for elements whose spans are at the sentence level (indexed by sentence)
# token_level_open_info: contains info for elements within the sentence that open at/before a given token (indexed by sentence and token)
# token_level_close_info: contains info for elements within the sentence that close at/after a given token (indexed by sentence and token)
# metadata: dictionary of the file metadata from the attributes of the 'text' element
def get_xml_info(xml_filepath, as_string=False):
	# get xml data
	if as_string:
		xml_data = xml_filepath.strip().split("\n")
	else:
		xml_data = open(xml_filepath, 'r').readlines()

	# initialize indexies 
	sentence_index = 0
	token_index = 0
	tag_id = 0
	last_token_index = 0
	# initialize dictionaries to store element information
	sentence_level_info = {}
	token_level_open_info = {}
	token_level_close_info = {}
	# initalize variable to help tracking state of location in the xml data
	currently_open_tags = []
	in_sentence = False
	trailing_s_close = False

	# iterate over the lines of the xml file
	for line in xml_data:

		# if we are at a line with an xml element
		if (line[0] == '<'):
			
			# skip encoding attribute and close tag for text; these lines need no annotation as they can be assumed upon reconstruction
			if (line[:5] == '<?xml' or line[:6] == '</text'):
				continue
			# retrieve the metadata for the file if we are at 'text' element, move the next line
			elif (line[:5] == '<text'):
				metadata = get_metadata(line)
				continue
			# if we are at an 's' element, mark that we are now inside a sentence, move to the next line
			elif (line[:3] == '<s ' or line[:3] == '<s>'):
				in_sentence = True
				trailing_s_close = False # probably ?
				continue

			# if we are at the end of a sentence, mark that are not inside a sentence, mark that we may be seeing tags that trail the sentence,
			# move the sentence index to the following sentence, reset the token index to 0 for the new sentence (but remember the last token 
			# index for the last sentence), move to next line
			elif (line[:4] == '</s>' or line[:4] == '</s '):
				in_sentence = False
				trailing_s_close = True
				sentence_index += 1
				last_sentence_length = token_index
				token_index = 0
				continue

			# if we are at a close tag
			elif (line[:2] == '</'):

				# find the sentence index and token index for the previous token
				if (token_index == 0):
					temp_sentence_index = sentence_index - 1
					temp_token_index = last_sentence_length - 1
				else:
					temp_sentence_index = sentence_index
					temp_token_index = token_index - 1

				# get most recent tag to be opened (last addition to currently_open_tags)
				recent_tag = currently_open_tags[-1]
				# remove most recent tag from currently_open_tags because it's being closed now
				currently_open_tags = currently_open_tags[:-1]

				# fill in end location of the tag
				recent_tag['end'] = { 'sentence_index' : temp_sentence_index, 'token_index' : temp_token_index }
				# get start index location for open tag of the closing tag's element
				start_sentence_index = recent_tag['start']['sentence_index']
				start_token_token = recent_tag['start']['token_index']

				# update end location in sentence_level_info and/or token_level_info
				if (not in_sentence):
					tag_index = 0
					for tag in sentence_level_info[start_sentence_index]:
						# if we are at the tag that is currently being closed
						if (tag['tag_id'] == recent_tag['tag_id']):
							# replace the tag info with the tag with the end location added
							sentence_level_info[start_sentence_index][tag_index] = recent_tag
							# if the tag is in a sentence start is after the sentence stop location, we are potentially in a 'trailing s' position, the last currently open tag (if there is one) is not starting at next s tag, we are a tag trailing the previous sentence 
							# -> remove the tag and append it to end of previous sentence :: and currently open tag not starting at next s tag
							if (recent_tag['start']['sentence_index'] > recent_tag['end']['sentence_index'] and recent_tag['trailing_s'] and 
							(len(currently_open_tags) == 0 or currently_open_tags[-1]['start']['sentence_index']  != recent_tag['start']['sentence_index']) ): #if (recent_tag['start']['sentence_index'] == recent_tag['end']['sentence_index'] and recent_tag['start']['token_index'] == recent_tag['end']['token_index'] and recent_tag['start']['token_index'] != 0): # this is rough
								# the tag is in a sentence trailing position - move it's tag info to the previous sentnce
								# it's location information will signal that it is a sentence tailing tag
								sentence_level_info[start_sentence_index].remove(recent_tag)
								if (start_sentence_index-1) in sentence_level_info:
									sentence_level_info[start_sentence_index-1].append(recent_tag)
								else:
									sentence_level_info[start_sentence_index-1] = [recent_tag]
							else:
								# if a tag closes we are no longer directly trailing the previous sentence
								trailing_s_close = False
							# we can break once we have found and updated the tag that is closing
							break
						tag_index += 1

				# update end location in token_level_open_info and token_level_close_info
				if (in_sentence):
					tag_index = 0
					for tag in token_level_open_info[start_sentence_index][start_token_token]:
						# if we are at the opening entry of the tag that is currently being closed
						if (tag['tag_id'] == recent_tag['tag_id']):
							# replace the tag info with the tag with the end location added
							token_level_open_info[start_sentence_index][start_token_token][tag_index] = recent_tag
							# we can break once we have found and updated the opening entry of the tag that is closing
							break
						tag_index += 1
					
					# append the tag info for the closing tag to the list for the token that it follows
					if (temp_sentence_index in token_level_close_info):
						if (temp_token_index in token_level_close_info[temp_sentence_index]):
							token_level_close_info[temp_sentence_index][temp_token_index].append(recent_tag)
						else:
							token_level_close_info[temp_sentence_index][temp_token_index] = [recent_tag]
					else:
						token_level_close_info[temp_sentence_index] = {}
						token_level_close_info[temp_sentence_index][temp_token_index] = [recent_tag]

			else:
				# we are at an open tag
				# create tag representation
				tag = create_tag_representation(line, sentence_index, token_index, tag_id, trailing_s_close)
				tag_id += 1
				# add tag to currently_open_tags
				currently_open_tags.append(tag)

				# if we are outside a sentence, put tag info in sentence_level_info
				if (not in_sentence):
					if (sentence_index in sentence_level_info):
						sentence_level_info[sentence_index].append(tag)
					else:
						sentence_level_info[sentence_index] = [tag]
					
				# if we are in a sentence, put tag info in token_level_open_info
				if (in_sentence):
					if (sentence_index in token_level_open_info):
						if (token_index in token_level_open_info[sentence_index]):
							token_level_open_info[sentence_index][token_index].append(tag)
						else:
							token_level_open_info[sentence_index][token_index] = [tag]
					else:
						token_level_open_info[sentence_index] = {}
						token_level_open_info[sentence_index][token_index] = [tag]

		else:
			# we are at a line with a token entry, increase the token_index
			token_index += 1

	return sentence_level_info, token_level_open_info, token_level_close_info, metadata


# get a dictionary of the metadata for the file from the 'text' element's attributes
# we need to add skipping the title
def get_metadata(text_tag):
	metadata = {}
	text_tag = text_tag.strip()[5:-1] # removing <text >
	tags = shlex.split(text_tag)
	for tag in tags:
		if ('=' in tag):
			(attr, val) = tag.split('=', 1)
			metadata[attr] = val
	return metadata


# creates a dictionary object with the element's location, attribute, and identifiying information
def create_tag_representation(tag_line, sentence_index, token_index, tag_id, trailing_s_close):
	
	tag = {}
	line_list = shlex.split(tag_line.strip()[1:-1]) # remove <>
	tag['tag_id'] = tag_id
	tag['trailing_s'] = trailing_s_close
	tag['text'] = line_list[0]
	tag['attrib'] = {}
	tag['start'] = { 'sentence_index' : sentence_index, 'token_index' : token_index }
	tag['end'] = {}
	line_list = line_list[1:]
	for attrib in line_list:
		key, value = attrib.split('=', 1)
		tag['attrib'][key] = re.sub('=', ':::', value)

	return tag

# for a given conllu file and the dictionaries storing the corresponding xml tag info, 
# add the xml annotations to the conllu data and write a new conllu file
def update_conllu_file(conllu_filepath, updated_conllu_filepath, sentence_level_info, token_level_open_info, token_level_close_info, metadata,
					   as_string=False, add_metadata=True):

	if as_string:
		conllu_data = parse(conllu_filepath)
	else:
		conllu_data = parse(open(conllu_filepath, 'r').read())
	sentence_index = 0
	xml_token_index = 0
	conllu_token_index = 0

	for sentence in conllu_data:

		# add sentence level comments
		if (sentence_index in sentence_level_info):
			newpar_num = 0
			trailing_xml = ''
			for tag in sentence_level_info[sentence_index]:
				if (tag['start']['sentence_index'] != sentence_index):
					# the start index of the tag is not the same as the ccurrent sentence index, we are a trailing tag
					# create string of attributes from the tag object
					attrib_string = ''
					for attrib in tag['attrib']: 
						attrib_string += attrib + ':::' + '\"' + tag['attrib'][attrib] + '\" '
					attrib_string = attrib_string[:-1]
					# add open tag to trailing xml string
					if (len(tag['attrib']) > 0):
						trailing_xml += '<' + tag['text'] + ' ' + attrib_string + '>'
					else:
						trailing_xml += '<' + tag['text'] + '>'
					# add close tag to trailing xml string
					trailing_xml += '</' + tag['text'] + '>'
				else:
					# the tag spans 1 or more sentences

					# create string of attributes from the tag object
					attrib_string = ''
					for attrib in tag['attrib']: 
						attrib_string += attrib + ':::' + '\"' + tag['attrib'][attrib] + '\" '

					# find the num of sentences the tag spans
					tag_span = str(tag['end']['sentence_index'] - tag['start']['sentence_index'] + 1) 

					# add sentence comment to metadta
					tag_string = ''
					if (len(tag['attrib']) > 0):
						tag_string = tag['text'] + ' ' + attrib_string + '(' + tag_span + ' s)'
					else:
						tag_string = tag['text'] + ' (' + tag_span + ' s)'

					if("newpar" in conllu_data[sentence_index].metadata):
						conllu_data[sentence_index].metadata["newpar"] += " | " + tag_string
					else:	
						conllu_data[sentence_index].metadata["newpar"] = tag_string

			# add close tags to trailing_xml
			if (trailing_xml != ''):
				conllu_data[sentence_index].metadata['trailing_xml'] = trailing_xml

		for token in sentence:

			if(type(token['id']) is not int):
				# skip multiword token line entries
				conllu_token_index += 1
				continue

			# add open tags to token
			if (sentence_index in token_level_open_info and xml_token_index in token_level_open_info[sentence_index]):
				for tag in token_level_open_info[sentence_index][xml_token_index]:
					
					# create string of attributes from the tag object
					attrib_string = ''
					for attrib in tag['attrib']: 
						attrib_string += attrib + ':::' + '\"' + tag['attrib'][attrib] + '\" '
					attrib_string = attrib_string[:-1]

					# create tag string
					tag_string = ''
					if (len(tag['attrib']) > 0):
						tag_string = '<' + tag['text'] + ' ' + attrib_string + '>'
					else:
						tag_string = '<' + tag['text'] + '>'

					# add tag string to conllu token's misc. column
					if (type(conllu_data[sentence_index][conllu_token_index]['misc']) is dict):
						if ('XML' in conllu_data[sentence_index][conllu_token_index]['misc']):
							conllu_data[sentence_index][conllu_token_index]['misc']['XML'] += tag_string
						else:
							conllu_data[sentence_index][conllu_token_index]['misc']['XML'] = tag_string
					else:
						conllu_data[sentence_index][conllu_token_index]['misc'] = {'XML': tag_string}

			# add close tags to token
			if (sentence_index in token_level_close_info and xml_token_index in token_level_close_info[sentence_index]):
				for tag in token_level_close_info[sentence_index][xml_token_index]:
					# make tag string
					tag_string = '</' + tag['text'] + '>'
					# add close tag string to conllu token's misc. column
					if (type(conllu_data[sentence_index][conllu_token_index]['misc']) is dict):
						if ('XML' in conllu_data[sentence_index][conllu_token_index]['misc']):
							conllu_data[sentence_index][conllu_token_index]['misc']['XML'] += tag_string
						else:
							conllu_data[sentence_index][conllu_token_index]['misc']['XML'] = tag_string
					else:
						conllu_data[sentence_index][conllu_token_index]['misc'] = {'XML': tag_string}

			# increment token indexies
			xml_token_index += 1
			conllu_token_index += 1

		# reset the indexies for the next sentence
		sentence_index += 1
		xml_token_index = 0
		conllu_token_index = 0

	contents = "\n\n".join([sentence.serialize().strip() for sentence in conllu_data])
	new_content = ''
	if add_metadata:
		for key in metadata:
			new_content += '# meta::' + key + ' = ' + metadata[key] + '\n'
	contents = new_content + contents
	if as_string:
		return contents
	else:
		with open(updated_conllu_filepath, 'w') as f:
			# write updated conllu data to new file
			f.write(contents)


# for a given conllu file with xml annotations, use the annotations to reconstruct the xml and write it to a file
def reconstruct_xml(conllu_filepath, updated_xml_filepath):
	conllu_data = parse(open(conllu_filepath, 'r').read())
	xml_string = ''
	trailing_xml = ''
	#xml_string = '<?xml version=\'1.0\' encoding=\'utf8\'?>\n'
	
	# add metadata from the top of the conllu file to the 'text' attribute
	metadata = {}
	conllu_lines = open(conllu_filepath, 'r').readlines()
	for line in conllu_lines:
		if (line[:8] == '# meta::'):
			(key, value) = line[8:].split(' = ', 1)
			metadata[key] = value[:-1]
	metadata_str = '<text '
	for attr in metadata:
		metadata_str += attr + '=' + '\"' + metadata[attr] + '\"' + ' '
	metadata_str = metadata_str[:-1] + '>'
	xml_string += metadata_str + '\n'

	# initialize span tracker for open tags
	span_tracker = []

	for sentence in conllu_data:

		# close tags that have finished their span

		# close last sentence tag
		if(sentence.metadata['sent_id'][-2:] != '-1'):
			xml_string += '</s>\n'

		# add trailing xml from previous sentence
		if (trailing_xml != ''):
			trailing_xml_lines = re.findall('<.*?>', trailing_xml) # split things contained in angle brackets
			for line in trailing_xml_lines:
				xml_string += re.sub(':::', '=', line) + '\n'
		
		# empty trailing xml string
		trailing_xml = ''

		# add close for tags that have finished their span and remove them from the tracker, decrement all spans by 1
		span_index = 0
		for span in span_tracker:
			if(span[1] - 1 == 0):
				xml_string += '</' + span[0] + '>\n'
			span_tracker[span_index][1] -= 1
			span_index += 1
		span_tracker = [span for span in span_tracker if span[1] > 0]

		# parse sentence level comment tags
		sentence_tag = ''
		for entry in sentence.metadata:

			if (entry == 's_type'):
				sentence_tag = '<s type=\"' + sentence.metadata[entry] + '\">' 

			# construct open tag line from the comment
			if ('newpar' in entry):
				for element_info in sentence.metadata[entry].split(" | "):
					entry_info = shlex.split(element_info)
					tag_name = entry_info[0]
					tag_span = entry_info[-2][1:]
					entry_tag = '<' + tag_name
					tag_attribs = entry_info[1:-2]
					for attribute in tag_attribs:
						key, value = attribute.split(":::", 1)
						entry_tag += ' ' + key + '=\"' + value + '\"'
					entry_tag += '>'

					# store the number of sentences that the tag spans
					if(int(tag_span) != 0):
						span_tracker = [[tag_name, int(tag_span)]] + span_tracker

					# add the line to the xml string
					xml_string += re.sub(':::', '=', entry_tag) + '\n'

					# close the tag if it doesn't span any sentences
					if(int(tag_span) == 0):
						xml_string += '</' + tag_name + '>\n'

			if(entry == 'trailing_xml'):
				trailing_xml = sentence.metadata[entry]

		# close any tags whose span has reached 0 and remove them from the tracker
		for span in span_tracker:
			if(span[1] == 0):
				xml_string += '</' + span[0] + '>\n'
		span_tracker = [span for span in span_tracker if span[1] > 0]

		# add the sentence tag
		if (sentence_tag == ''):
			sentence_tag = '<s>'
		xml_string += sentence_tag + '\n'

		for token in sentence:
			if (type(token['id']) is not int):
				# skip multiword token entries
				continue
			# if token has xml annotation
			if (type(token['misc']) is dict and 'XML' in token['misc']):
				# if there are open tags
				if (token['misc']['XML'][:2] != '</'):
					# grab them and add them to the xml string
					split_index = token['misc']['XML'].find('</')
					if (split_index != -1):
						open_tags = token['misc']['XML'][:split_index]
						open_tags = re.sub(':::', '=', open_tags)
					else:
						open_tags = token['misc']['XML']
					open_tags = re.sub(':::', '=', open_tags)
					open_tags_list = re.findall(r'<.*?>', open_tags)
					for tag in open_tags_list:
						xml_string += tag + '\n'
				# then add the token to the xml string
				xml_string += token['form'] + '\t' + token['xpos'] + '\t' + token['lemma'] + '\n'
				# if there are close tags
				if ('</' in token['misc']['XML']):
					# grab them and add them
					if (token['misc']['XML'][:2] == '</'):
						close_tags = token['misc']['XML']
					else:
						close_tags = token['misc']['XML'][split_index:]
					close_tags_list = re.findall(r'<.*?>', close_tags)
					for tag in close_tags_list:
						xml_string += tag + '\n'
			else:
				# if there is no xml annotation, just add the token line
				xml_string += token['form'] + '\t' + token['xpos'] + '\t' + token['lemma'] + '\n'
	
	xml_string += '</s>\n'

	# add xml that directly trails the sentence close
	if (trailing_xml != ''):
		trailing_xml_lines = re.findall('<.*?>', trailing_xml) # split things contained in angle brackets
		for line in trailing_xml_lines:
			xml_string += re.sub(':::', '=', line) + '\n'

	# close any remaining tags
	for span in span_tracker:
		xml_string += '</' + span[0] + '>\n'

	xml_string += '</text>'

	with open(updated_xml_filepath, 'w') as f:
		# write updated conllu data to new file
		f.write(xml_string)			

	return

# returns 1 if the 2 files match and returns 0 if they do not
def evaluate_reconstruction(original_xml_filepath, reconstructed_xml_filepath):
	original_xml = open(original_xml_filepath, 'r').readlines()
	reconstructed_xml = open(reconstructed_xml_filepath, 'r').readlines()
	line_number = 0
	sucess = True
	for original_line in original_xml:
		if ( original_line == reconstructed_xml[line_number] or (original_line[0] != '<' and html.unescape(original_line) == reconstructed_xml[line_number])): # or original_line == reconstructed_xml[line_number] html.unescape(original_line)
			line_number += 1
		else:
			print(reconstructed_xml_filepath)
			print("The files did not match starting on line nubmer " + str(line_number))
			print("Original: " + original_line)
			print("Reconstruction: " + reconstructed_xml[line_number])
			sucess = False
			break
	if sucess:
		return 1
	else:
		return 0


def add_xml(conllu, xml):
	sentence_level_info, token_level_open_info, token_level_close_info, metadata = get_xml_info(xml, as_string=True)
	conllu = update_conllu_file(conllu, "", sentence_level_info, token_level_open_info, token_level_close_info, metadata,
								as_string=True, add_metadata=False)
	return conllu


if __name__ == "__main__":
	xml_files = os.listdir(REDUC_DIR)
	done_right = 0
	# iterate through xml files
	for xml_file in xml_files:
		conllu_file = xml_file.replace('.xml', '.conllu')
		xml_filepath = REDUC_DIR + xml_file
		conllu_filepath = ORIG_DIR + conllu_file
		# extract the info for the xml tags
		sentence_level_info, token_level_open_info, token_level_close_info, metadata = get_xml_info(xml_filepath)
		updated_conllu_filepath = UPD_DIR + conllu_file
		updated_xml_filepath = RECON_DIR + xml_file
		# update the conllu file with xml annotations
		update_conllu_file(conllu_filepath, updated_conllu_filepath, sentence_level_info, token_level_open_info, token_level_close_info, metadata)
		# reconstruct the xml file from the annotations in the conllu file
		reconstruct_xml(updated_conllu_filepath, updated_xml_filepath)
		# check that the resonstruction matches the original
		done_right += evaluate_reconstruction(xml_filepath, updated_xml_filepath)
	print('Number of proper reconstructions out of 5117:')
	print(done_right)