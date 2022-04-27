from email.mime import base
from pathlib import Path
import os
import time

def script(base_name: str, dump: int):
	return f"""#!/bin/bash

#SBATCH --job-name dump{dump:05d}
#SBATCH --output output/dump{dump:05d}.out
#SBATCH -c 4
#SBATCH -p kipac,normal,hns
#SBATCH -t 4:00:00

cd /oak/stanford/orgs/kipac/users/pizza/MSM/msm-data-interpreter/
cargo run --example analyze_bose_star "{base_name}" {dump} --release"""



def gen_scripts(base_name: str, ndumps: int):

	actual_base = base_name.split("/")[-1].replace("-", "")
	print(f"Making dir at batches/{actual_base}-batches/")
	Path(f"batches/{actual_base}-batches/").mkdir(parents=True, exist_ok=True)

	for dump in range(ndumps+1):

		content = script(base_name, dump)

		print(f"file at batches/{actual_base}-batches/submit_{dump:05d}.sbatch")
		file = open(f"batches/{actual_base}-batches/submit_{dump:05d}.sbatch", "w")
		file.write(content)
		file.close()

def run_scripts(base_name, ndumps):

	actual_base = base_name.split("/")[-1].replace("-", "")

	for dump in range(ndumps+1):
		time.sleep(1)
		os.system(f"sbatch batches/{actual_base}-batches/submit_{dump:05d}.sbatch")

if __name__ == "__main__":
    	
	# Generate scripts
	base_name = "../rust/sim_data/bose-star-512"
	ndumps = 0
	gen_scripts(base_name, ndumps)

	# Run scripts
	run_scripts(base_name, ndumps)

