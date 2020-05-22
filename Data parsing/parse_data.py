
import pickle

def only_short_name_entity(p):
	for i in range(len(p)):
		element = p[i]

		if "<name type=" in element:
			if "</name>" in p[i + 2]:
				continue
			else:
				return False

	return True


def has_propn(p):
	for x in p:
		if "<name type" in x:
			return True
	return False


def process_sentence(p):	
	
	semafor = False
	propn_buffer = ""

	solution = []

	for i in range(len(p)):

		line = p[i]
		

		#get rid of useless meta lines however don't ignore the <name line because we need it's data
		if line  and ("name" in line or line[0] != "<"):
			
			if "<name type" in line:
				semafor = True
				
				#store they type of name into the buffer to be used in the next line
				propn_buffer = line
				continue

			if "</name" in line:
				semafor = False
				continue

			split_line = line.split()
			
			if semafor:

				#this means this is a propn and the class is stored in propn_buffer
				solution.append((split_line[0], split_line[1], split_line[2], propn_buffer.replace("<name type=", "").replace(">", "").replace('"', "")))
			else:
				#here we do all the work for normal lines
				solution.append((split_line[0], split_line[1], split_line[2]))

	return solution

def construct_whole_sentence(solution):
	s = ""
	
	for i in range(len(solution) -1):
		element = solution[i]

		if not solution[i+ 1][0].isalnum():
			s += element[0]
		else:
			s+= element[0] + " "

	s += solution[-1][0]
	return s

if __name__ == "__main__":
	file = open("tagget_named_entities.vert", "r")

	file_string = file.read()
	file_string_split = file_string.split("</s>")
	


	
	count_all = 0
	count_short = 0
	count_propn = 0

	final_dict = dict()

	for sentence in file_string_split:
		
		sentence_lines = sentence.split("\n")

		if len(sentence_lines) > 1 and only_short_name_entity(sentence_lines):
			a = process_sentence(sentence_lines)
			s = construct_whole_sentence(a)

			final_dict[s] = a
			

			print(s)
			print("-------------------")
			for key in a:
				print(key)
			print("-------------------")
			

		










		#some simple statistics
		if only_short_name_entity(sentence_lines):
			count_short += 1
		count_all += 1
		
		if has_propn(sentence_lines):
			count_propn += 1



	print(count_all)
	print(count_short)
	print(count_propn)
		
		




	file = open("parsed_tagged_named_enteties.pickle", "wb")
	pickle.dump(final_dict, file)
	file.close()