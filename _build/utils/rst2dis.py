import io, os, re
from glob import glob
from six import iteritems
from collections import defaultdict
from argparse import ArgumentParser

class Unit:

	def __init__(self):
		self.id = 0
		self.text = ""
		self.right = -1
		self.left = 100000
		self.relname = ""
		self.parent = 0
		self.type = "edu"
		self.children = []
		self.satnuc = "N"
		self.depth = -1
		self.unary = False

	def to_dis(self):
		text = "\n"
		cat = "( Root " if self.parent == 0 else "( Nucleus "
		if self.satnuc == "S":
			cat = "( Satellite "
		if self.type == "edu":
			span = "(leaf " + str(self.id)+")"
		else:
			span = "(span " + str(self.left) + " " + str(self.right) + ")"
		if self.parent == 0:
			rel2par = " "
		else:
			rel2par = " (rel2par "+self.relname+")"
		if self.type == "edu":
			text = " (text _!"+self.text+"_!) )\n"
		for child in self.children:
			text += child.to_dis()
		indent = self.depth * "  "
		dis = indent + cat + span + rel2par + text
		if self.type != "edu":
			dis += indent + ")"
		if self.type != "edu":
			dis += "\n"
		return dis



	def __repr__(self):
		return str(self.id) +"["+str(self.left)+"-" +str(self.right)+"]"+" (" + self.type+")" +": " + self.text


def get_descendants(node,all_nodes,child_dict):
	descendants = set([])
	if node.id in child_dict:
		children = child_dict[node.id]
		for child_id in children:
			descendants.add(child_id)
			child = all_nodes[child_id]
			descendants.update(get_descendants(child,all_nodes,child_dict))
	return descendants


def assign_depth(node,rank):
	node.depth = rank
	for child in node.children:
		assign_depth(child,rank+1)


def assign_span(node,descendants,all_units):
	desc_units = [all_units[d] for d in descendants[node.id]]
	to_remove = []  # List for direct satellite children of this node, which do not participate in span
	if node.id == 37:
		a=6
	for u in desc_units:
		if u in node.children:
			if u.satnuc == "S":
				to_remove.append(u)
	for u in to_remove:
		desc_units.remove(u)
		if u.id in descendants:
			for desc_id in descendants[u.id]:
				desc_units.remove(all_units[desc_id])
	left_most_desc = min(desc_units,key=lambda x:x.left)
	right_most_desc = max(desc_units,key=lambda x:x.right)
	node.left = left_most_desc.left
	node.right = right_most_desc.right


def sort_children(units):
	for uid in units:
		u = units[uid]
		u.children = list(set(u.children))  # Ensure unique children
		u.children.sort(key=lambda x:(x.left,-x.right))

def rewire_parents(units):
	for uid in units:
		u = units[uid]
		if uid == 60:
			a=3
		if u.satnuc == "S" and units[u.parent].relname == "span":
			parent = units[u.parent]
			span = parent.parent
			parent.children.remove(u)
			units[span].children.append(u)
			u.parent = span

def rst2dis(rs3_string, remove_unary=True, binarize=True):

	multinuc_rels = set([])
	satellite_rels = set([])
	units = {}
	root = None
	max_right = 0
	lines = rs3_string.strip().split("\n")

	for i, line in enumerate(lines):
		if "<rel" in line:
			m = re.search(r'<rel name="([^"]+)" type="([^"]+)"/>',line)
			if m is not None:
				if m.group(2) == "multinuc":
					multinuc_rels.add(m.group(1))
				else:
					satellite_rels.add(m.group(1))
		elif "<segment" in line:
			m = re.search(r'<segment id="([^"]+)" parent="([^"]+)" relname="([^"]+)">(.*?)</segment>',line)
			if m is not None:
				u = Unit()
				u.id = int(m.group(1))
				u.parent = int(m.group(2))
				u.relname = m.group(3)
				u.text = m.group(4)
				u.type = "edu"
				u.left = u.id
				u.right = u.id
				if u.id > max_right:
					max_right = u.id
				if u.relname in satellite_rels:
					if u.relname not in multinuc_rels:
						u.satnuc = "S"
					else: # Ambiguous relation name
						u.satnuc = "ambig"

				units[u.id] = u
			else:
				raise IOError("Invalid input at line " + str(i) + ": " + line)
		elif "<group" in line:
			m = re.search(r'<group id="([^"]+)" type="([^"]+)"',line)
			if m is not None:
				u = Unit()
				u.id = int(m.group(1))
				u.type = m.group(2)
				if u.id == 60:
					a=4
				m = re.search(r'parent="([^"]+)" relname="([^"]+)"',line)  # For non-root nodes
				if m is not None:
					u.parent = int(m.group(1))
					u.relname = m.group(2)
					if u.relname in satellite_rels:
						if u.relname not in multinuc_rels:
							u.satnuc = "S"
						else: # Ambiguous relation name
							u.satnuc = "ambig"
				else:
					u.depth = 0
					root = u
				units[u.id] = u
			else:
				raise IOError("Invalid input at line " + str(i) + ": " + line)


	# Find children
	children = defaultdict(set)
	for uid in units:
		u = units[uid]
		children[u.parent].add(u.id)
		if u.parent != 0:
			units[u.parent].children.append(u)

	# Check if root has satellites - if so, add a span above it
	for child in root.children:
		if child.satnuc != "N":
			new_root = Unit()
			new_root.id = -1
			new_root.left = 1
			new_root.right = max_right
			new_root.children.append(root)
			new_root.type = "span"
			new_root.depth = 0
			children[new_root.id].add(root.id)
			root.parent = new_root.id
			root.relname = "span"
			units[-1] = new_root
			root = new_root
			break

	# Disambiguate multinuc rels
	for uid in units:
		u = units[uid]
		if u.satnuc == "ambig":
			if units[u.parent].type != "multinuc":
				u.satnuc = "S"
			else:  # Child of a multinuc with an ambiguous multinuc-possible relation
				# Since the multinuc rel must appear multiple times, we assume it has max frequency in the multinuc
				siblings = units[u.parent].children
				votes = defaultdict(int)
				for sib in siblings:
					if sib.relname in multinuc_rels:
						votes[sib.relname] += 1
				if u.relname == max(votes,key=lambda x: votes[x]):
					u.satnuc = "N"
				else:
					u.satnuc = "S"

	# Get recursive descendant list
	descendants = defaultdict(set)
	for uid in units:
		u = units[uid]
		descendants[u.id] = get_descendants(u,units,children)


	# Compute depth
	assign_depth(root,0)

	# Compute span
	for uid in units:
		u = units[uid]
		if u.type != "edu":
			assign_span(u,descendants,units)

	# Rewire parents - satellites of a span should have the span as a parent
	rewire_parents(units)

	# Flag and remove unary branches
	if remove_unary:
		for uid in units:
			u = units[uid]
			if len(u.children)==1:
				u.unary = True
				if u.parent != 0:
					u.children[0].parent = u.parent
					units[u.parent].children.append(u.children[0])
					units[u.parent].children.remove(u)


		units = dict((uid,u) for uid,u in iteritems(units) if not units[uid].unary)

	sort_children(units)

	# Binarize
	if binarize:
		ids = list(units.keys())
		for uid in ids:
			a=3
			if len(units[uid].children) > 2 and units[uid].type=="multinuc":
				new_unit = units[uid]
				while len(new_unit.children) > 2:
					rest = new_unit.children[1:]  # Children to binarize, after the first child
					u = Unit()  # Make a new dummy parent unit for the right branch of the n-ary multinuc
					u.id = max(units.keys()) + 1
					u.parent = new_unit.id
					new_unit.children.append(u)
					u.relname = rest[0].relname
					u.type = "multinuc"
					u.left = rest[0].left
					u.right = rest[-1].right
					new_unit = u
					units[new_unit.id] = new_unit  # Add the new unit to the units dict
					for child in rest:  # rewire parentage of remaining right-hand children
						units[child.parent].children.remove(child)
						child.parent = new_unit.id
						new_unit.children.append(child)

	assign_depth(root,0)

	sort_children(units)
	assign_depth(root,0)

	return root.to_dis()


if __name__ == "__main__":
	p = ArgumentParser()
	p.add_argument("files",help="glob pattern of files to convert")

	opts = p.parse_args()

	files = glob(opts.files)

	for file_ in files:
		rs3 = io.open(file_,encoding="utf8").read()
		dis = rst2dis(rs3)
		outfile = os.path.basename(file_).replace(".rs3","") + ".dis"

		with io.open(outfile,'w',encoding="utf8",newline="\n") as f:
			f.write(dis)

