from stat import S_ENFMT
import numpy as np
from glob import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Creates an instance of Dataset that gathers a particular data dump out of all simulations.
class DumpDataset(Dataset):
    def __init__(self, dumps):
        self.real_paths = dumps

    def __len__(self):
        return len(self.real_paths)

    def __getitem__(self, idx):

        psi = np.load(self.real_paths[idx]) + \
            1j*np.load(f"{self.real_paths[idx].replace('real', 'imag')}")


        return torch.from_numpy(psi)



# def load_complex(file: str):
#     # Load Data
#     data = np.load(file)
#     # Construct and return complex array
#     return data["real"] + 1j*data["imag"]


# def dump_complex(array: np.array, path: str): 
#     np.savez(
#         path,
#         real = np.real(array),
#         imag = np.imag(array)
#     )


# def average_dump(args):#sim_base_name: str, dump: int, K: int, S: int):
#     sim_base_name, dump, K, S = args

#     # Define buffers
#     size = np.ones(K, dtype=int)*S
#     #print(size)
#     psi = np.zeros(size) * (0 + 0j)
#     psi2 = np.zeros(size) * (0 + 0j)
#     psik = np.zeros(size) * (0 + 0j)
#     psik2 = np.zeros(size) * (0 + 0j)

#     # Debug check
#     #print(psi.shape)
#     assert psi.shape == (S, S, S), "wrong shape"

#     sims = glob(f"{sim_base_name}-stream*")
#     #print(sims)
#     for sim in sims:

#         # Construct location of wavefunction
#         dump_path = f"{sim}/psi_{dump:05d}"
#         #print(dump_path)
        
#         # Load wavefunction data
#         data = load_complex(dump_path)[:,:,:,0]
#         #print(data.shape)

#         # Add to buffers
#         psi += data
#         psi2 += data * data
#         psik += np.fft.fftn(data)
#         psik2 += np.fft.fftn(data) * np.fft.fftn(data)

#     # Average
#     psi /= len(sims)
#     psi2 /= len(sims)
#     psik /= len(sims)
#     psik2 /= len(sims)

#     path1 = f"{sim_base_name}-combined/psi_{dump}"
#     path2 = f"{sim_base_name}-combined/psi2_{dump}"
#     path3 = f"{sim_base_name}-combined/psik_{dump}"
#     path4 = f"{sim_base_name}-combined/psik2_{dump}"

#     dump_complex(psi, path1)
#     dump_complex(psi2, path2)
#     dump_complex(psik, path3)
#     dump_complex(psik2, path4)


# def pqdm_analyze_sims(sim_base_name: str,ndumps: int, K: int, S: int):

#     from pqdm.processes import pqdm

#     dumps = np.arange(ndumps)+1
#     args = [(sim_base_name, dump, K, S) for dump in dumps]
#     pqdm(args, average_dump, 16)

def torch_analyze_sims(
    sim_name: str,
    sim_data_dir: str,
    ndumps: int,
    K: int,
    S: int
):
    print("Begin analysis with torch")

    size = np.ones(K, dtype=int)*S
    while len(size) < 4:
        size = np.append(size, 1)

    # Define 
    psi = torch.zeros(tuple(s for s in size)).to(device) * (0 + 0j)
    psi2 = torch.zeros(tuple(s for s in size)).to(device)
    psik = torch.zeros(tuple(s for s in size)).to(device) * (0 + 0j)
    psik2 = torch.zeros(tuple(s for s in size)).to(device)


    # Iterate through each dump
    it = trange(ndumps)
    for dump in it:

        # Define dataset containing same dump from every dataset
        real_dump_names = np.sort(glob(f"{sim_data_dir}/{sim_name}-stream*/psi_{dump:05d}_real"))

        # Create dataloader
        dataset = DumpDataset(real_dump_names)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

        # Iterate over each psi in the dataset
        for (stream, psi_) in enumerate(data_loader):

            it.set_description(f"{sim_name}-stream{stream:05d}")
            sys.stdout.flush()

            # Send to device and get rid of batch_size=1 dimension
            psi_ = psi_.to(device)[0]
            assert (np.array(psi_.shape) == size).all(), f"wrong shape of {psi_.shape} vs expected {size}"

            # Add 1) wave function, 2) its mod squared, 3) its fft, 4) the fft's mod squared.
            psi += psi_
            psi2 += torch.abs(psi_)**2
            psik += torch.fft.fftn(psi_)
            psik2 += torch.abs(torch.fft.fftn(psi_))**2
        
        # Average
        psi /= dataset.__len__()
        psi2 /= dataset.__len__()
        psik /= dataset.__len__()
        psik2 /= dataset.__len__()

        # Create combined directory if it does not yet exist
        combined_path = f"{sim_data_dir}/{sim_name}-combined"
        os.makedirs(combined_path, exist_ok=True)

        # Save all 4 arrays to disk
        torch.save(  psi, f"{combined_path}/psi_{dump:05d}")
        torch.save( psi2, f"{combined_path}/psi2_{dump:05d}")
        torch.save( psik, f"{combined_path}/psik_{dump:05d}")
        torch.save(psik2, f"{combined_path}/psik2_{dump:05d}")
                

    

if __name__ == "__main__":

    sim_name = "gaussian-overdensity-512"
    sim_data_dir = "../rust/sim_data"
    ndumps = 100
    K = 3
    S = 512

    #for dump in np.arange(ndumps)+1:
        #average_dump((sim_base_name, dump, K, S))

    # pqdm_analyze_sims(sim_base_name, ndumps, K, S)

    torch_analyze_sims(sim_name, sim_data_dir, ndumps, K, S)


