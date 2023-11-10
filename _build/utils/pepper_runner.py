import threading,os,re,sys,io
from .nlp_helper import exec_via_temp
import platform

PY2 = sys.version_info[0] < 3

def compress_pepper_out(pepper_msg,full_log=False):
	empty_spans = 0

	if not PY2:
		pepper_msg = pepper_msg.decode("utf8")

	pepper_out = pepper_msg.replace("\r", "")
	lines = pepper_out.split("\n")
	for line in lines:
		if "no tokens contained in span" in line:
			empty_spans += 1

	# remove header
	pepper_out = re.sub(r'^.*?\*\*','',pepper_out, re.MULTILINE|re.DOTALL)
	pepper_out = re.sub(r'^.*step 1','',pepper_out, re.MULTILINE|re.DOTALL)
	# remove job description
	pepper_out = re.sub(r'-{4}-+.*?' + '-'*78 + '.*?\+','',pepper_out, re.MULTILINE|re.DOTALL)
	# remove job status messages
	pepper_out = re.sub(r'-+ pepper job status -+[^-]+-+','',pepper_out, re.MULTILINE|re.DOTALL)
	# remove empty span warnings
	pepper_out = re.sub(r'input file.*?span will be ignored!','',pepper_out)
	# remove meta tag messages
	pepper_out = re.sub(r"using meta tag '.*?'",'',pepper_out)
	# remove encoding messages
	pepper_out = re.sub(r"using input file encoding '.*?'",'',pepper_out)
	# remove footer
	pepper_out = re.sub(r"\*{4}\*+\n.*?\*{4}\*+",'',pepper_out, re.MULTILINE|re.DOTALL)
	pepper_out = re.sub(r"\n +\n",r'\n',pepper_out, re.MULTILINE|re.DOTALL)
	pepper_out = re.sub(r"\n+",r'\n',pepper_out, re.MULTILINE|re.DOTALL)

	# Get pepper messages
	messages = ""
	m = re.search(r'(Conversion ended[^\n\r]*)',pepper_out)
	if m is not None:
		messages += m.group(1)
	m = re.search(r'([^\n\r]*exception[^\n\r]*)',pepper_out)
	if m is not None:
		messages += m.group(1)
	m = re.search(r'([^\n\r]*\.java:[^\n\r]*)',pepper_out)
	if m is not None:
		messages += m.group(1)

	if not full_log:
		messages += "\n\n(In case of errors you can get verbose pepper output using the -v flag)"

	report = ""
	if empty_spans > 0:
		report += "i Pepper reports " + str(empty_spans) + " empty xml spans were ignored\n"
	report += "i Pepper says:\n\n"
	report += messages

	if full_log:
		report +="\n\nFull pepper output:\n\n"+pepper_msg

	return report


def runner(pepper_params,output):
	"""thread worker function"""
	if platform.system() == 'Windows':
		pepper_cmd = [os.path.abspath("utils" + os.sep + "pepper") + os.sep + "pepperStart.bat", "-p", "tempfilename"]
	else:
		pepper_cmd = [os.path.abspath("utils" + os.sep + "pepper") + os.sep + "pepperStart.sh", "-p", "tempfilename"]

	output[0] = exec_via_temp(pepper_params,pepper_cmd,os.path.abspath("utils" + os.sep + "pepper")+os.sep,False)
	return


def cycle_spinner(spinner):
	if spinner == "/":
		return "-"
	elif spinner == "-":
		return "\\"
	elif spinner == "\\":
		return "|"
	elif spinner == "|":
		return "/"


def run_pepper(pepper_params,full_log=False):
	# Open new thread for pepper so we don't lose control of the cli
	threads = []
	output = [""] # Placeholder variable to get output via modification by ref
	t = threading.Thread(target=runner,args=(pepper_params,output))
	threads.append(t)
	t.start()
	spinner = "/"
	while t.isAlive():
		spinner = cycle_spinner(spinner)
		sys.__stdout__.write("Pepper is working... " + spinner + "\r")
		t.join(1)
	sys.__stdout__.write(" " *30 + "\n")

	script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
	annis_out_dir = script_dir + ".." + os.sep + "target" + os.sep + "annis" + os.sep
	# Rename ANNIS namespaces
	ext = "annis"

	# Node annotation
	with io.open(annis_out_dir + "node_annotation." + ext,encoding="utf8",newline="\n") as f:
		node_annotation = f.read().replace("\r","")
		node_annotation = node_annotation.replace("default_ns\tkind","rst\tkind")
		node_annotation = node_annotation.replace("default_ns\ttype","rst\ttype")
		node_annotation = node_annotation.replace("default_ns\tsignal", "rst\tsignal")
		node_annotation = node_annotation.replace("default_ns\tCopyOf","dep\tCopyOf")
		node_annotation = node_annotation.replace("default_ns\tCorrectForm","dep\tCorrectForm")
		node_annotation = node_annotation.replace("default_ns\tclaws5","gum\tclaws5")
		node_annotation = node_annotation.replace("default_ns\ttok_func","gum\ttok_func")
		node_annotation = node_annotation.replace("default_ns","tei")
		node_annotation = node_annotation.replace("\trst\t","\tdefault_ns\t")  # revert default_ns for rst++
	with io.open(annis_out_dir + "node_annotation." + ext,"w",encoding="utf8",newline="\n") as f:
		f.write(node_annotation)

	# Edge annotation
	with io.open(annis_out_dir + "edge_annotation." + ext,encoding="utf8") as f:
		edge_annotation = f.read().replace("\r","")
		edge_annotation = edge_annotation.replace("default_ns\trelname","rst\trelname")
		edge_annotation = edge_annotation.replace("dep\tdeprel","dep\tfunc")
		edge_annotation = edge_annotation.replace("ptb\tfunc","const\tfunc")
		edge_annotation = edge_annotation.replace("default_ns\tfunc","dep\tfunc")
	with io.open(annis_out_dir + "edge_annotation." + ext,"w",encoding="utf8",newline="\n") as f:
		f.write(edge_annotation)

	# Component
	with io.open(annis_out_dir + "component." + ext,encoding="utf8") as f:
		component = f.read().replace("\r","")
		component = component.replace("dep\tud","ud\tud")
		component = component.replace("default_ns\thead","head\thead")
		component = component.replace("ud\tdep","dep\tdep")
	with io.open(annis_out_dir + "component." + ext,"w",encoding="utf8",newline="\n") as f:
		f.write(component)

	return compress_pepper_out(output[0],full_log)

