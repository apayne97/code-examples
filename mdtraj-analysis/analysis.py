"""
Contains functions for standardized analysis of trajectories.

"""

import mdtraj as md
import matplotlib.pyplot as plt
import simtk.unit as unit
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from simtk.openmm.app import PDBxFile, CharmmPsfFile
from math import pi
import numpy as np
import pandas as pd
import plotly as pt
import plotly.express as px
import plotly.graph_objects as go


## Next section has a few useful simple scripts
def pymol_to_mdtraj(pymol_string):
    """
    Simple function to convert a pymol-like residue string into mdtraj-like
    """
    x = pymol_string.replace('+', ' or residue ')
    y = x.replace('-', ' to ')
    final = f'residue {y}'
    return final


def get_chain_df_dict_from_large_chain_df_dict(chain_df_dict, single_selection):
    """
    Given a dictionary of dataframes by chain, this selects a subset of each dataframe
    based on the selection of interest.
    """
    new_chain_df_dict = {}
    chainids = range(len(chain_df_dict))
    chainletters = list(chain_df_dict.keys())

    dist0id = single_selection[0][0]
    dist0name = single_selection[0][1]
    dist1id = single_selection[1][0]
    dist1name = single_selection[1][1]
    time = chain_df_dict[chainletters[0]]['Time']

    for chainid in chainids:

        if chainid == 1:
            if dist0name == dist1name:
                continue
            dist0id = chainids[dist0id - 1]
            dist1id = chainids[dist1id - 1]

        dist0chainletter = chainletters[dist0id]
        dist1chainletter = chainletters[dist1id]

        if dist0name == dist1name and chainid == 0:
            series0name = f'{dist0chainletter}_{dist0name}'
            series1name = f'{dist1chainletter}_{dist1name}'
        else:
            series0name = f'{dist0name}'
            series1name = f'{dist1name}'

        series0 = chain_df_dict[dist0chainletter][dist0name]
        series1 = chain_df_dict[dist1chainletter][dist1name]

        new_chain_df_dict[chainletters[chainid]] = pd.DataFrame({'Time': time,
                                                                 series0name: series0,
                                                                 series1name: series1,
                                                                 })
    return new_chain_df_dict


def get_long_df_from_chain_df_dict(chain_df_dict, single_chain_columns, time, data_label = 'Data'):
    dfs = {}
    chains = list(chain_df_dict.keys())
    for chain in chains:
        df = pd.DataFrame(chain_df_dict[chain], columns=single_chain_columns)
        df['Time'] = time
        newdf = pd.melt(df, id_vars='Time')
        newdf.columns = ['Time', 'Label', data_label]
        newdf['Chain'] = chain
        dfs[chain] = newdf
    # return_df = pd.concat(dfs.values())
    return dfs


## RMSD Plotting Function for all the Selections
def get_and_plot_rmsds_single_plot(t, topo_dict, ref=None, precentered=False, ps_per_frame = 1000, rmsd_range=[0,5], title = 'RMSD', pdf_file='all_rmsds.pdf'):
    """
    Assumes all selections in the topo_dict are meant to be plotted on top of each other. Good for data exploration.
    """
    
    pdf = PdfPages(pdf_file)
    plt.figure()
    for label, topo in topo_dict.items():
        idx = t.topology.select(topo)
        print(label, topo)
        
        if not ref:
            ref = t
        
#         if ref_idx == 'protein':
#             ref_idx = t.topology.select('protein')
#         else:
#             ref_idx = idx

        rmsd_array = md.rmsd(target=t,
                             reference=ref,
                             frame=0,
                             atom_indices=idx,
                            precentered=precentered)

        ## Want to put this on the figure for comparison
        n_residues = t.atom_slice(idx).n_residues

        ## Get x and y values for plot
        rmsd_array_in_A = rmsd_array * 10
        time_in_ns = [x * ps_per_frame / 1000 for x in range(len(rmsd_array))]

        ## Plot the stuff
        # plt.figure() ## Commenting this out lets me plot multiple things on the same plot
        plt.plot(time_in_ns, rmsd_array_in_A, label=f'{label} ({n_residues} Residues)')
        
        #if label == 'Protein Backbone':
        total_rmsd = rmsd_array_in_A
            
    plt.xlabel('Time (ns)')
    plt.ylabel(f'RMSD (A)')
    plt.ylim(rmsd_range)
    plt.legend()
    plt.title(f'{title}')
    plt.tight_layout()
    pdf.savefig()

    pdf.close()
    
    return total_rmsd

# def get_per_residue_RMSD_avg(t, topo_dict):
    
## RMSD Plotting Function for all the Selections split by chain
def get_and_plot_rmsds_by_chain(t, topo_dict, ref=None, ps_per_frame=1000, title='RMSD by Chain', pdf_file='all_rmsds_by_chain.pdf'):
    pdf = PdfPages(pdf_file)
    
    if not ref:
        ref = t
    
    for label, topo in topo_dict.items():

        plt.figure()

        for chain, chainid in {'Chain A': 0, 'Chain B': 1}.items():
            topo_chain = f'{topo} and {chainid}'
            idx = t.topology.select(f'{topo} and chainid {chainid}')
            print(label, topo_chain, chain)

            rmsd_array = md.rmsd(target=t,
                                 reference=ref,
                                 frame=0,
                                 atom_indices=idx)

            ## Want to put this on the figure for comparison
            n_residues = t.atom_slice(idx).n_residues
            ## Get x and y values for plot
            rmsd_array_in_A = rmsd_array * 10
            time_in_ns = [x * ps_per_frame / 1000 for x in range(len(rmsd_array))]

            ## Plot the stuff
            plt.plot(time_in_ns, rmsd_array_in_A, label=f'{label} ({chain})')
        plt.xlabel('Time (ns)')
        plt.ylabel(f'{label} RMSD (A)')
        plt.ylim([0, 10])
        plt.legend()
        plt.title(f'{label} {title} ({n_residues} Residues)')
        plt.tight_layout()
        pdf.savefig()

    pdf.close()

def get_and_plot_rmsf_by_chain(t, title='RMSF by Chain', label_yaml_path = "../pymol-scripts/tmem175-RMSD-true-indexed.yaml", pdf_file='rmsf_by_chain.pdf'):
    chain_dict = {'Chain A': 0, 'Chain B': 1}
    chain_rmsf_dict = {}
    
    ## make figure larger for easy reading
    plt.figure(figsize=(8, 6), dpi=80)
    plt.ylim([0,10])
    for chain, chainid in chain_dict.items():
        idx = t.topology.select(f'protein and chainid {chainid} and name CA')
        subset = t.atom_slice(idx)

        rmsf = md.rmsf(target=t,
                       reference=None,
                       frame=0,
                       atom_indices=idx)

        resids = [id for id in range(30, subset.n_residues + 30)]

        ##Save as angstroms
        chain_rmsf_dict[chain] = rmsf * 10

        plt.plot(resids, rmsf * 10, label=chain)
    pdf = PdfPages(pdf_file)
    plt.title(title)
    plt.ylabel('C-alpha RMSF (Å)')
    plt.xlabel('Residue Number (30-477)')
    plt.xticks(resids[::10], rotation=90)
    
    ## add vertical helices labels
    ## load true helix indices:
    with open(label_yaml_path, 'r') as file: label_dict = yaml.safe_load(file)
    
    ## get only helices from label_dict
    helix_dict = {label: index for label, index in label_dict.items() if not 'Loop' in label}
    helix_dict
    
    for label, indices in helix_dict.items():
        ## get start and end of helix from file
        start, end = indices.split('-')

        ## they are still strings though, convert to integer
        idx = (int(start), int(end))

        ## just taking the average to place the text somewhere
        middle = (idx[0] + idx[1]) / 2 

        ## plot index
        plt.axvspan(idx[0], idx[1], zorder=0, alpha=0.2, color='blue')
        plt.text(x=middle, y=8, s=label, rotation='vertical')
    plt.tight_layout()
    plt.legend()
    pdf.savefig()
    pdf.close()

def get_total_rmsf(t):
    idx = t.topology.select(f'protein and name CA')
    subset = t.atom_slice(idx)

    rmsf = md.rmsf(target=t,
                   reference=None,
                   frame=0,
                   atom_indices=idx)
    total_rmsf = rmsf.sum()
    return total_rmsf
    
def compare_rmsd_to_each_other(t_dict, topo, frame):
    ## expects a dictionary that looks like this:
    ## t_dict = {1: [t1, 'Name'], 2: [t2: 'Name']}
    
    ps_per_frame = 1000
    
    def plot_rmsd(t1, t2, frame, label):
        idx = t1.topology.select(topo)
        rmsd_array = md.rmsd(target=t1,
                             reference=t2,
                             frame=frame,
                             atom_indices=idx)    
        ## Get x and y values for plot
        rmsd_array_in_A = rmsd_array * 10
        time_in_ns = [x * ps_per_frame / 1000 for x in range(len(rmsd_array))]

        ## Plot the stuff
        # plt.figure() ## Commenting this out lets me plot multiple things on the same plot
        plt.plot(time_in_ns, rmsd_array_in_A, label = label)
    
    ## feels like a janky way to do this but the point is 
    ## to compare each trajectory to the specific frame of both trajectories
    for a,b in [(1,1), (1,2), (2,1), (2, 2)]:
        t1 = t_dict[a][0]
        t2 = t_dict[b][0]
        label = f'{t_dict[a][1]} to {t_dict[b][1]}'
        plot_rmsd(t1, t2, frame=frame, label=label)
    
    
    plt.xlabel('Time (ns)')
    plt.ylabel(f'RMSD (A)')
    plt.legend()
    plt.tight_layout()
    
def get_ile_core_dihedrals(t):
    """
    Plot the core isoleucines 46 and 271 2 sidechain dihedral angles
    """
    
    labels = ['46 chain A', '46 chain B', '271 chain A', '271 chain B']

    for res in [46, 271]:
        # Chi 1
        plt.figure()
        plt.title('Chi 1')
        dihedrals = t.atom_slice(t.topology.select(f'residue {res} and protein'))
        chi1s = md.compute_chi1(dihedrals)
        plt.plot(t.time, chi1s[1])
        plt.ylim(-pi, pi)
        plt.legend(labels=[f'{res} chain A', f'{res} chain B'])

        # Chi 2
        plt.figure()
        plt.title('Chi 2')
        dihedrals = t.atom_slice(t.topology.select(f'residue {res} and protein'))
        chi2s = md.compute_chi2(dihedrals)
        plt.plot(t.time, chi2s[1])
        plt.ylim(-pi, pi)
        plt.xticks(t.time[::5])
        plt.legend(labels=[f'{res} chain A', f'{res} chain B'])
        plt.tight_layout()

def construct_chi12_df(t, selection):

    ## First get the chi1 and chi2 dihedrals for the given selection
    df = pd.DataFrame()
    resSlice = t.atom_slice(t.topology.select(selection))
    chi1s = md.compute_chi1(resSlice)
    chi2s = md.compute_chi2(resSlice)
    reslist = [f'{str(residue)}_{residue.chain.index}' for residue in resSlice.topology.residues]

    ## convert the output of the compute method into a dataframe
    chi1df = pd.DataFrame(chi1s[1] * 180 / 3.14, columns = reslist)
    chi2df = pd.DataFrame(chi2s[1] * 180 / 3.14, columns = reslist)

    ## combine the dataframes into one
    combined_df = pd.DataFrame()
    for i in range(len(reslist)):
        res = reslist[i]
        chi1s = chi1df[res]
        chi2s = chi2df[res]
        newdf = pd.DataFrame({'Time': chi1df.index, 'Res': res, 'Chi1': chi1s, 'Chi2': chi2s})
        combined_df = combined_df.append(newdf)
    return combined_df


def plot_dihedrals(t, selection):
    combined_df = construct_chi12_df(t, selection)
    fig = px.scatter(combined_df, x='Chi1', y='Chi2', facet_col='Res', color='Time', facet_col_wrap=2)
    fig.update_yaxes(range=[-185, 185])
    fig.update_xaxes(range=[-185, 185])
    fig.show()

def get_activation_network_contacts(t):
    dist_labels = {'HIS57_0 to ASP279_0': '(residue 57 and chainid 0) or (residue 279 and chainid 0)',
                   'HIS57_0 to GLU282_0': '(residue 57 and chainid 0) or (residue 282 and chainid 0)',
                   'HIS57_0 to GLU282_1': '(residue 57 and chainid 0) or (residue 282 and chainid 1)',
                   'HIS57_1 to ASP279_1': '(residue 57 and chainid 1) or (residue 279 and chainid 1)',
                   'HIS57_1 to GLU282_1': '(residue 57 and chainid 1) or (residue 282 and chainid 1)',
                   'HIS57_1 to GLU282_0': '(residue 57 and chainid 1) or (residue 282 and chainid 0)',
                   }
    df = pd.DataFrame({'Time': t.time})
    for label, string in dist_labels.items():
        newt = t.atom_slice(t.topology.select(string))
        distances = md.compute_contacts(newt, np.array([[0,1]]))[0] * 10
        df[label] = distances
    return df


def plot_activation_network_dist_against_each_other(df, title, filepath):
    ## Currently a very bespoke way to analyze activation network distances

    ## Assuming we can make this a variable it would be slightly more manageable
    dist_dict = {
        'Chain A': {'x': 'HIS57_0 to ASP279_0', 'y': 'HIS57_0 to GLU282_1', 'row': 1, 'col': 1},
        'Chain B': {'x': 'HIS57_1 to ASP279_1', 'y': 'HIS57_1 to GLU282_0', 'row': 2, 'col': 1},
    }
    fig = pt.subplots.make_subplots(rows=2, cols=1,
                        subplot_titles=[*dist_dict])

    for chain, chaindict in dist_dict.items():
        x = chaindict['x']
        y = chaindict['y']
        row = chaindict['row']
        col = chaindict['col']
        fig.add_trace(
            px.scatter(df,
                       x=x,
                       y=y,
                       color='Time',
                       ).data[0],
            row=row, col=col
        )
        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text=y, row=row, col=col)
    fig.update_layout(height=800, width=800, title_text=title)
    fig.update_xaxes(range=[2, 15])
    fig.update_yaxes(range=[2, 15])
    fig.show()
    fig.write_image(filepath)




def get_mdtraj_idx_array(t, dist_array):
    """
    Given a numpy array that looks like this:
    (
    ((XX, chainid), (YY, chainid)),
    ((XX, chainid), (YY, chainid)),
    )
    """
    dist_label_dict = {}
    
    ## Not super necessary but nice to get the names:
    protein = t.atom_slice(t.topology.select('chainid 0 or chainid 1'))
    chainA = t.atom_slice(t.topology.select('chainid 0'))
    n_residues = chainA.topology.n_residues
    
    ## this will returned
    names = []
    
    ## these will be return as a dict
    chainA_idx_list = []
    chainB_idx_list = []
    
    ## for each pair of residues in the dist_array
    for res1_set, res2_set in dist_array:
        
        ## get the naive res numbers
        res1_number = res1_set[0]
        res2_number = res2_set[0]
        
        ## for chain A, chain B
        for chain in [0,1]:
            
            ## set res1
            res1_chainid = chain
            
            
            ## if it is zero, then we can take the chainid's as is
            if res1_chainid == 0:
                res2_chainid = res2_set[1]
                
                ## have to account for missing residues
                res1id = res1_number - 30
            
            ## otherwise, we have to flip the chainid
            elif res1_chainid == 1:
                res2_chainid = int(not res2_set[1])
                
                ## we also have to add n_residues to res1 to give it chain 0's number
                res1id = res1_number - 30 + n_residues
            
            ## now we have set res2's chainid, so we can set the resid
            if res2_chainid == 0:
                res2id = res2_number - 30
            elif res2_chainid == 1:
                res2id = res2_number - 30 + n_residues
            
            print('RES NUMBER, CHAIN:', res1_number, res1_chainid, res2_number, res2_chainid)
            
            print('RES ID:', res1id, res2id)
            
            
            ## get the residues!
            res1 = protein.topology.residue(res1id)
            res2 = protein.topology.residue(res2id)
            
            if res1_chainid == 0:
                ## this is a nice check just to make sure the chain index is the one we think we are using
                ## we will save the name based on the first residue being in chain 0
                string = f'{res1}_{res1.chain.index} to {res2}_{res2.chain.index}'
                names.append(string)
                
                chainA_idx_list.append([res1id, res2id])
                
                
            elif res1_chainid == 1:
                chainB_idx_list.append([res1id, res2id])
    
    ## now we combine the two lists
    
    idx_dict = {'ChainA': chainA_idx_list, 'ChainB': chainB_idx_list}
    
    return names, idx_dict

def get_df_from_idx_dict(t, names, idx_dict, return_long_df=True):
    dfs = {}
    for chain, idx_array in idx_dict.items():
        contacts, residue_pairs = md.compute_contacts(t, contacts=idx_array)
        contacts_Angstroms = contacts * 10
        df = pd.DataFrame(contacts_Angstroms, columns=names)
        
        if return_long_df:
            df['Time'] = t.time
            newdf = pd.melt(df, id_vars='Time')
            newdf.columns = ['Time', 'Label', 'Minimum Heavy Atom Distance (Å)']
            newdf['Chain'] = chain
        else:
            newdf = pd.DataFrame(df, columns=names)
            newdf['Time'] = t.time
            
        dfs[chain] = newdf
    if return_long_df:
        return_obj = pd.concat(dfs.values())
    else:
        print('return chain dict')
        # return_df = pd.merge(dfs['ChainA'],dfs['ChainB'], on="Time")
        return_obj = dfs
    return return_obj

# def plot_dist_from_long_df(long_df, xrange=[0,15], yrange=[0,1], binsize=0.5, title='Title', filepath='test.pdf', height=800, width=800):
#     fig = px.histogram(long_df,
#                        x='Minimum Heavy Atom Distance (Å)',
#                        facet_col='Label',
#                        color='Chain',
#                       histnorm='probability'
#                       )
#     fig.update_traces(xbins=dict( # bins used for histogram
#         start=xrange[0],
#         end=xrange[1],
#         size=binsize
#     ))
#     fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
#     fig.update_layout(barmode='overlay')
#     fig.update_traces(opacity=0.5)
#     fig.update_xaxes(range=xrange)
#     fig.update_yaxes(range=yrange)
#     fig.update_layout(height=height, width=width, title_text=title)
# #     fig.show()
# #     fig.write_image(filepath)
#     return fig
def plot_dist_from_long_df(long_df):
    fig = px.histogram(long_df,
                       x='Minimum Heavy Atom Distance (Å)',
                       facet_col='Label',
                       color='Chain',
                      histnorm='probability'
                      )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

def plot_dist_scatterplot(long_df, title, selection):
    ## this version accepts a long dataframe
    chains = list(set(long_df['Chain']))
    
#     chains = [0,1]
    print(chains)
    fig = pt.subplots.make_subplots(rows=2, cols=1,
                        subplot_titles=chains,
                                   )
    for dist1, dist2 in selection:
        dist1_chainid = dist1[0]
        dist1_dist = dist1[1]
        dist2_chainid = dist2[0]
        dist2_dist = dist2[1]
        
        for chain in chains:
            
            print(chain)
            if chain == 'ChainA':
                row = 1
                dist1_chain = chains[dist1_chainid]
                dist2_chain = chains[dist2_chainid]
            elif chain == 'ChainB':
                row = 2
                dist1_chain = chains[dist1_chainid-1]
                dist2_chain = chains[dist2_chainid-1]
            print(dist1_chain, dist1_dist, dist2_chain, dist2_dist)
            
    

            x = list(long_df[(long_df['Label'] == dist1_dist) & (long_df['Chain'] == dist1_chain)]['Minimum Heavy Atom Distance (Å)'])
            y = list(long_df[(long_df['Label'] == dist2_dist) & (long_df['Chain'] == dist2_chain)]['Minimum Heavy Atom Distance (Å)'])
            time = list(long_df[(long_df['Label'] == dist1_dist) & (long_df['Chain'] == dist1_chain)]['Time'])
            dist1_label = f'{dist1_dist} {dist1_chain}'
            dist2_label = f'{dist2_dist} {dist2_chain}'
            newdf = pd.DataFrame({dist1_label: x, dist2_label: y, 'Time': time} )
            print(newdf)
            
            print(row)
            col = 1
            fig.add_trace(
                px.scatter(newdf,
                            x=dist1_label,
                           y=dist2_label,
                           color='Time',
                           ).data[0],
                row=row, col=col
            )
            fig.update_xaxes(title_text=dist1_label, row=row, col=col)
            fig.update_yaxes(title_text=dist2_label, row=row, col=col)
    fig.update_layout(height=800, width=800, title_text=title)
    fig.update_xaxes(range=[2, 15])
    fig.update_yaxes(range=[2, 15])
#         fig.show()
#         fig.write_image(filepath)
    return fig


def plot_distances_both_chains(selection_dict):
    chains = list(selection_dict.keys())
    print(chains)
    n_cols = 2
    row = 1
    fig = pt.subplots.make_subplots(rows=row, cols=n_cols,
                                    subplot_titles=chains,
                                    #                                     column_widths=[0.5,0.5]
                                    # shared_yaxes=True
                                    )
    for idx in range(n_cols):
        col = idx + 1
        print(n_cols, row, col)
        chain = chains[idx]
        df = selection_dict[chain]
        print(chain)
        x_label = df.columns[1]
        y_label = df.columns[2]
        print(x_label)

        fig.add_trace(
            px.scatter(df,
                       x=x_label,
                       y=y_label,
                       color='Time',
                       ).data[0],
            row=row, col=col
        )
        #         fig.add_trace(
        #             px.density_contour(df,
        #                               x=x_label,
        #                               y=y_label,
        #                               ).data[0],
        #             row=row, col=col
        #         )
        #         fig.update_traces(contours_coloring="fill",
        #                           #contours_showlabels = True
        #                           colorscale='Blues',
        #                          )
        fig.update_xaxes(
            # title_text=f'{x_label} Minimum Heavy Atom Distance (Å)',
                         #                          range=[2,12],
                         row=row,
                         col=col,
                                                  constrain="domain",
                         )
        fig.update_yaxes(
            # title_text=f'{y_label} Minimum Heavy Atom Distance (Å)',
                         #                             range=[2,12],
                                                     scaleanchor = "x",
                                                     scaleratio = 1,
                         row=row,
                         col=col,
                         )
    #     fig.show()
    return fig

def plot_distances_across_chains(selection_dict):
    chains = list(selection_dict.keys())
    chain = chains[0]
    df = selection_dict[chain]
    x_label = df.columns[1]
    y_label = df.columns[2]
    fig = pt.subplots.make_subplots(rows=1, cols=1)
    fig.add_trace(
        px.scatter(
            df,
            x=x_label,
            y=y_label,
            color='Time',
        ).data[0]

    )
    # fig = px.scatter(df,
    #                x=x_label,
    #                y=y_label,
    #                color='Time',
    #                )
    fig.update_xaxes(
        # title_text=f'{x_label} Minimum Heavy Atom Distance (Å)',
                     constrain="domain",
                    )
    fig.update_yaxes(
        # title_text=f'{y_label} Minimum Heavy Atom Distance (Å)',
                        scaleanchor = "x",
                        scaleratio = 1,
                    )
#     fig.show()
    return fig

def get_total_sasa_by_chain(t, selection):
    sasa_dict = {}
    for chain in [0,1]:
        newt = t.atom_slice(t.topology.select(f'({selection}) and chainid {chain}'))
        sasa = md.shrake_rupley(newt)
        total_sasa = sasa.sum(axis=1)
        sasa_dict[chain] = total_sasa
    df = pd.DataFrame({'ChainA': sasa_dict[0], 'ChainB': sasa_dict[1], 'Time': t.time})
    long_df = pd.melt(df, id_vars='Time')
    long_df.columns = ['Time', 'Chain', 'Total SASA']
    return long_df

def sys_dict_data_iterator(func, sys_dict, data_name, *args, **kwargs):
    for name, info in sys_dict.items():
        print(name)
        t = info['traj']
        sys_dict[name][data_name] = func(t, *args, **kwargs)
    return sys_dict

def sys_dict_clones_data_iterator(func, sys_dict, data_name, *args, **kwargs):
    for system in sys_dict.keys():
        for name, info in system.items():
            print(name)
            t = info['traj']
            sys_dict[name][data_name] = func(t, *args, **kwargs)
        return sys_dict

def sys_dict_plot_iterator(func, sys_dict, data_name, selector=False, pdf_dir=False, pdf_tag=False, update_layout_kwargs=False, update_xaxes_kwargs=False, update_yaxes_kwargs=False, update_traces_kwargs=False,**kwargs):
    for name, info in sys_dict.items():
        print(name)
        t = info['traj']
        data = info[data_name]

        if selector:
            data = data[data['Label'].isin(selector)]

        if info.get('Plot Title'):
            title = info['Plot Title']
        else:
            title = f'{info["Title"]}, {info["State"]} Equilibrated with {info["Equilibration"]} and run for {info["Length"]}ns'
        fig = func(data, **kwargs)
        fig.update_layout(title=title)
        if update_layout_kwargs:
            fig.update_layout(**update_layout_kwargs)
        
        if update_xaxes_kwargs:
            fig.update_xaxes(**update_xaxes_kwargs)
        
        if update_yaxes_kwargs:
            fig.update_yaxes(**update_yaxes_kwargs)

        if update_traces_kwargs:
            fig.update_traces(**update_traces_kwargs)
        
        print('Showing figure...')
        fig.show()
        
        if pdf_dir and pdf_tag:
            filepath = f'{pdf_dir}/{name}_{pdf_tag}.pdf'
            print('Saving figure...')
            fig.write_image(filepath)