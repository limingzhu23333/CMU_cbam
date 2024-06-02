import os

def list_files(directory):
	files=os.listdir(directory)
	return files
	
	
def write_to_txt(file_list,output_file):
	with open(output_file,"w") as f:
		for file_name in file_list:
			f.write(file_name + '\n')


if __name__ == "__main__":
	input_directory = "./Validation/Image"
	output_file ="./Validation_file.txt"
	
	file_list = list_files(input_directory)
	write_to_txt(file_list,output_file)
