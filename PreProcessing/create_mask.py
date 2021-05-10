import os
import subprocess

from subprocess import CalledProcessError

base = os.path.expanduser('~')
dataset = os.path.join(base, 'Datasets', 'ImageCLEF', 'Dataset')
ctscans = os.listdir(dataset)

maskdir = os.path.join(base, 'Datasets', 'ImageCLEF', 'masks')

def create_mask_dir():
	try:
		os.makedirs(maskdir)
	except FileExistsError as err:
		files = os.listdir(maskdir)
		print(f'Directory: {maskdir} already exists. Deleting all files from it.')
	
		for file in files:
			del_path = os.path.join(maskdir, file)
			os.remove(del_path)

def log_error(log_file, source):
	err = f'ERROR creating mask for: {source}'
	log_file.write(err)

def create_mask(log_file, source, destination):
	try:
		subprocess.run(['lungmask', source, destination], check=True)
		print(f'Created: {destination}')
	except CalledProcessError as err:
		log_error(log_file, source)


def main():
	create_mask_dir()

	with open('Create_masks.log', 'w') as fr:
		for scan in ctscans:
			source = os.path.join(dataset, scan)
			
			scan = scan.split('.')[0]
			newname = f'{scan}_mask.nii.gz'
			destination = os.path.join(maskdir, newname)

			create_mask(fr, source, destination)

if __name__=='__main__':
	main()

