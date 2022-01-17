import os.path

import yaml
import pandas as pd
import mdtraj as md
from simtk.openmm.app import CharmmPsfFile


def load_sysdf(simulation_table_path="simulations.yaml"):
    """
    loads the simulations yaml file and returns a transformed dataframe
    :param simulation_table_path:
    :return:
    """
    with open(simulation_table_path, 'r') as file: sim_dict = yaml.safe_load(file)
    sysdf = pd.DataFrame(sim_dict).T
    return sysdf


def import_traj(sysdir, trajFile='trajectory.dcd'):
    """
    Assumes trajectory strided to 1ns / frame and aligned with md.superpose()
    """
    aPath = sysdir + 'analysis/'
    # oPath= sysdir + 'output/' ## not being used right now
    iPath = sysdir + 'input/'
    trajPath = aPath + trajFile
    topPath = iPath + 'step5_input.psf'

    ## load dcd into mdtraj object
    t = md.load_dcd(trajPath,
                    top=topPath,
                    stride=1)

    ## get topology from psf file using openmm
    psf = CharmmPsfFile(topPath)

    ## convert topology to mdtraj topology and save to mdtraj object
    t.topology = t.topology.from_openmm(psf.topology)

    return t


def import_traj_from_munged(traj_path, psf_path, pdb_path):
    """
    Assumes trajectory strided to 1ns / frame and aligned with md.superpose()
    """

    ## load dcd into mdtraj object
    t = md.load_dcd(traj_path,
                    top=psf_path,
                    stride=1)
    
    pdb = md.load(pdb_path)

    ## get topology from psf file using openmm
    psf = CharmmPsfFile(psf_path)

    ## convert topology to mdtraj topology and save to mdtraj object
    t.topology = t.topology.from_openmm(psf.topology)
    pdb.topology = pdb.topology.from_openmm(psf.topology)

    return t, pdb


def import_systems(sysdf):
    """
    Takes dataframe of systems and returns a dictionary with all the info from the dataframe plus the md.trajectory object
    :param sysdf:
    :return:
    """
    sys_dict = {}
    for idx in sysdf.index:
        sysinfo = dict(sysdf.loc[idx])
        name = sysinfo["Title"]
        path = sysinfo["Path"]
        trajFile = sysinfo["TrajFile"]
        print(f'Loading {name} from {path + trajFile}')
        traj = import_traj(sysdir=path, trajFile=trajFile)
        sysinfo['traj'] = traj
        sys_dict[name] = sysinfo
    return sys_dict


def import_systems_from_munged(sys_names, traj_prefixes, traj_file_extension, n_clones, munged_path):
    assert type(sys_names) == list
    assert type(n_clones) == int
    assert type(traj_prefixes) == list

    with open('munged_simulations.yaml', 'r') as file:
        sim_dict = yaml.safe_load(file)
    print(sim_dict)

    with open('prefix_dict.yaml', 'r') as file:
        prefix_dict = yaml.safe_load(file)
    print(prefix_dict)

    sys_dict = {}
    for system in sys_names:
        sys_info = sim_dict[system]
        # clone_dict = {}
        for idx in range(n_clones):
            for traj_prefix in traj_prefixes:
                clone_info = {}
                clone_info.update(sys_info)
                clone = f'{traj_prefix}_clone{idx:02d}'

                traj_path = f'{munged_path}{system}/{clone}.{traj_file_extension}'
                psf_path = f'{munged_path}{system}/step5_input.psf'
                pdb_path = f'{munged_path}{system}/step5_input.pdb'
                
                assert os.path.exists(traj_path), f'{traj_path} does not exist!'
                assert os.path.exists(psf_path), f'{psf_path} does not exist!'

                clone_info['Length'] = prefix_dict[traj_prefix]['Length']

                full_name = f'{system}_{clone}'
                clone_info['Title'] = full_name
                
                
                ## Save mdtraj trajectory object of the loaded pdb path
                pdb_path = f'{munged_path}{system}/step5_input.pdb'
                
                print(f'Loading {full_name} from {traj_path}')
                
                traj, pdb = import_traj_from_munged(traj_path, psf_path, pdb_path)
                clone_info['Input PDB'] = pdb
                clone_info['traj'] = traj

                sys_dict[full_name] = clone_info

    return sys_dict
