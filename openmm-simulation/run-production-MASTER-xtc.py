## load System from XML Files
# import sys
# from sys import stdout
import mdtraj as md
# import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
from simtk.openmm import LangevinMiddleIntegrator
from simtk.openmm import CustomExternalForce
from simtk.openmm.app import CharmmPsfFile, PDBFile, PDBxFile, CharmmParameterSet
# from openmmtools.integrators import LangevinIntegrator
import simtk.unit as unit
import os, time, yaml, bz2, argparse, logging

## Use argparse for i/o and timelength info
parser = argparse.ArgumentParser()
parser.add_argument('-pFile', dest='pFile', required=True)
parser.add_argument('-oDir', dest='oDir', required=True)
parser.add_argument('-logFile', dest='logFile', required=True)
parser.add_argument('-iDir', dest='iDir', required=False)
args = parser.parse_args()

# now we will Create and configure logger
logging.basicConfig(filename=args.logFile,
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Let us Create an object
logger = logging.getLogger()

# Now we are going to Set the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

## Load Params from yaml file
logger.info(f'Loading parameters from {args.pFile}...')
with open(args.pFile) as f:
    pDict = yaml.safe_load(f)
    oDir = args.oDir

## Check to make sure we are going to get a positions file from somewhere
if not pDict.get('iStatePath') and not args.iDir and not pDict.get('iPDBPath') and not pDict.get('chkFile'):
    logger.error(
        f'\n\tNo positions to use!\n'
        f'\tIt appears that {args.pFile} does not contain\n'
        f'\teither an "iStatePath" or a "iPDBPath" and you have not\n'
        f'\tspecified an input directory.\n'
    )
    raise Exception

logger.info(f'Setting parameters from {args.pFile} with units...')
unitsDict = {}
unitsDict['hydrogen_mass'] = pDict['hydrogen_mass'] * unit.amu
unitsDict['temperature'] = pDict['temperature'] * unit.kelvin
unitsDict['friction'] = pDict['friction'] / unit.picoseconds
unitsDict['time_step'] = pDict['time_step'] * unit.picoseconds
unitsDict['pressure'] = pDict['pressure'] * unit.bar
unitsDict['surface_tension'] = pDict['surface_tension']  ## units are complicated but its usually zero anyways
for name, value in unitsDict.items():
    logger.info(f'\t{name}: {value}')

if not os.path.isdir(oDir):
    logger.info(f'Making output directory: "{oDir}"')
    os.mkdir(oDir)
logger.info(f'Setting output files to be:')
oDict = {}
for name, f in pDict['outputFiles'].items():
    filename = os.path.join(oDir, f)
    logger.info(f'\t{filename}')
    oDict[name] = filename

## Load PSF File - will need topology from it whether we create a system or not
logger.info(f'Loading psf from {pDict["psfPath"]}')
psf = CharmmPsfFile(pDict['psfPath'])

if args.iDir:
    ## If given a directory and the associated files, we don't have to create a system
    iDir = args.iDir
    logger.info(f'Getting input files from {iDir}:')
    iDict = {}
    for name, f in pDict['inputFiles'].items():
        filename = os.path.join(iDir, f)
        logger.info(f'\t{filename}')
        iDict[name] = filename

    if iDict.get('chkFile'):
        chkFile = iDict['chkFile']

    else:
        iStateFile = iDict['iStateFile']
        iSystemFile = iDict['iSystemFile']
        iIntegratorFile = iDict['iIntegratorFile']

        ## Deserialize system
        logger.info(f'Getting system from {iSystemFile}')
        with bz2.open(iSystemFile, 'rb') as infile:
            system = mm.XmlSerializer.deserialize(infile.read().decode())

        ## Deserialize Integrator
        logger.info(f'Getting integrator from {iIntegratorFile}')
        with bz2.open(iIntegratorFile, 'rb') as infile:
            integrator = mm.XmlSerializer.deserialize(infile.read().decode())

if args.iDir:
    if iDict.get('chkFile'):
        create_system = True
    else:
        create_system = False
else:
    create_system = True

if create_system == True:
    ## Load CHARMM Params
    logger.info(f'Loading CHARMM Parameters: {pDict["param_paths"]}')
    params = CharmmParameterSet(*pDict['param_paths'])

    ## Setting up the system using the psf file
    ## get box size from sysinfo.dat
    import json

    with open(pDict['boxsizePath']) as file:
        data = json.load(file)
        x, y, z = map(float, data['dimensions'][:3]) * unit.angstroms
    logger.info(f'Setting box size x: {x}, y: {y}, z:{z}')
    psf.setBox(x, y, z)  
    ## Set System Params
    nonbonded_method = app.PME
    constraints = app.HBonds

    logger.info(f'Creating system...')
    system = psf.createSystem(params,
                              nonbondedMethod=nonbonded_method,
                              constraints=constraints,
                              removeCMMotion=False,
                              hydrogenMass=unitsDict['hydrogen_mass'],
                              )

    ## Set Integrator and Barostat
    logger.info(f'Creating integrator..')
    integrator = LangevinMiddleIntegrator(unitsDict['temperature'],
                                          unitsDict['friction'],
                                          unitsDict['time_step'])

    logger.info(f'Setting barostat...')
    barostat = mm.MonteCarloMembraneBarostat(unitsDict['pressure'],
                                             unitsDict['surface_tension'],
                                             unitsDict['temperature'],
                                             mm.MonteCarloMembraneBarostat.XYIsotropic,
                                             mm.MonteCarloMembraneBarostat.ZFree
                                             )
    barostat.setFrequency(50)  ## for some reason the __init__ won't accept it as an argument, but this works
    ## the default is 25 timesteps, i've set it for 50
    system.addForce(barostat)

    if pDict.get('iStatePath'):
        iStateFile = pDict['iStatePath']
    else:
        iStateFile = False

## Create Simulation
# set up the platform
logger.info(f'Setting up platform...')
if pDict.get('platform'):
    platform_name = pDict['platform']
else:
    platform_name = 'CUDA'
platform = mm.Platform.getPlatformByName(platform_name)

logger.info('\tUsing mixed precision!')
platform.setPropertyDefaultValue('Precision', 'mixed')

logger.info(f'Creating simulation object...')
sim = app.Simulation(psf.topology,
                     system=system,
                     integrator=integrator,
                     platform=platform,
                     )
if args.iDir:
    if iDict.get('chkFile'):
        logger.info(f'Using checkpoint file {chkFile}')
        with open(chkFile, 'rb') as f:
            sim.context.loadCheckpoint(f.read())

logger.info(f'Confirming platform is: {sim.context.getPlatform().getName()}')

## Setting state
if iStateFile:
    logger.info(f'Getting state from {iStateFile}')
    with bz2.open(iStateFile, 'rb') as infile:
        state = mm.XmlSerializer.deserialize(infile.read().decode())

    logger.info(f'Setting state from {state}...')
    sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    sim.context.setPositions(state.getPositions())
    sim.context.setVelocities(state.getVelocities())

elif pDict.get('iPDBPath'):
    logger.info(f'Getting pdb from {pDict["iPDBPath"]}...')
    pdb = PDBFile(pDict["iPDBPath"])

    logger.info(f'Setting positions and velocities...')
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(unitsDict['temperature'])
    

## adding restraints if they exist
if pDict.get('restraint_info'):
    logger.info(f'Adding restraints...')
    from simtk.openmm import CustomExternalForce
    # Add Backbone Restraints
    restraint_type = pDict['restraint_info']['restraint_type']
    restraint_k = pDict['restraint_info']['restraint_k'] * unit.kilojoules_per_mole / unit.nanometer**2 # kJ / mol / nm^2

    logger.info(f'\tAdding {restraint_type} restraints with a k constant of {restraint_k}')

    ## Create Formula for CustomExternal Forces
    posresPROT = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2;')
    posresPROT.addGlobalParameter('k', restraint_k)
    posresPROT.addPerParticleParameter('x0')
    posresPROT.addPerParticleParameter('y0')
    posresPROT.addPerParticleParameter('z0')

    crd = sim.context.getState(getPositions=True).getPositions()
    system = sim.context.getSystem()

    atoms = [atom for atom in sim.topology.atoms()]

    for i, atom_crd in enumerate(crd):
        if atoms[i].name in ('CA','C', 'N') and atoms[i].residue.chain.id in ('PROA', 'PROB'):
            print(i, atom_crd)
            posresPROT.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
    force_id = system.addForce(posresPROT)

    logger.info('\tReinitializing context')
    sim.context.reinitialize(preserveState=True)
    
    forces = [force for force in sim.context.getSystem().getForces()]
    logger.info(f'\tSuccessfully added force: {forces[force_id]}')

## Adjust time parameters
current_time = sim.context.getState().getTime() / unit.nanoseconds
total_simulation_time = pDict['nsteps'] * unitsDict['time_step'] / unit.nanoseconds
simulation_time = total_simulation_time - current_time
steps_left = round(simulation_time * unit.nanoseconds / unitsDict['time_step'])

traj_freq_in_time = pDict["traj_freq"] * unitsDict['time_step'] / unit.nanoseconds
report_freq_in_time = pDict["report_freq"] * unitsDict['time_step'] / unit.nanoseconds
chk_freq_in_time = pDict["chk_freq"] * unitsDict['time_step'] / unit.nanoseconds

logger.info(f'\t{steps_left:.0f} steps, {simulation_time:.0f} ns, will be run for a total simulation time of {total_simulation_time:.3f} ns \n'
            f'\tSaving a frame to {oDict["oTrajFile"]} every {pDict["traj_freq"]} step(s), or every {traj_freq_in_time:.3f} ns \n'
            f'\tWriting checkpoint file every {pDict["chk_freq"]} step(s), or every {chk_freq_in_time:.3f} ns \n'
            f'\tWriting state info to {oDict["oLogFile"]} every {pDict["report_freq"]} step(s), or every {report_freq_in_time:.3f} ns \n'
            )

## write limited state information to standard out:
sim.reporters.append(
    app.StateDataReporter(
        oDict['oLogFile'],
        reportInterval=pDict['report_freq'],
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=steps_left,
        separator="\t",
    )
)

## Write to checkpoint files regularly:
sim.reporters.append(app.CheckpointReporter(
    file=oDict['oCheckFile'],
    reportInterval=pDict['chk_freq']
)
)

# Write out the trajectory

# sim.reporters.append(md.reporters.DCDReporter(
#     file=oDict['oTrajFile'],
#     reportInterval=pDict['traj_freq']
# )
# )

sim.reporters.append(md.reporters.XTCReporter(
    file=oDict['oTrajFile'],
    reportInterval=pDict['traj_freq']
))


# Run NPT dynamics
logger.info("Running dynamics in the NPT ensemble...")
initial_time = time.time()
sim.step(steps_left)
elapsed_time = (time.time() - initial_time) * unit.seconds
logger.info(
    f'\tEquilibration took {elapsed_time / unit.seconds:.3f} s at {simulation_time / elapsed_time * unit.day:.3f} ns/day)')
# % (elapsed_time / unit.seconds, simulation_time / unit.nanoseconds, simulation_time / elapsed_time * unit.day / unit.nanoseconds))

# Save and serialize the final state
logger.info("Serializing state to %s" % oDict['oStateFile'])
state = sim.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True,
    getForces=True
)

with bz2.open(oDict['oStateFile'], "wt") as outfile:
    xml = mm.XmlSerializer.serialize(state)
    outfile.write(xml)

# Save the final state as a PDBx File
logger.info("Saving final state as %s" % oDict['oPDBFile'])
with open(oDict['oPDBFile'], "wt") as outfile:
    PDBxFile.writeFile(
        sim.topology,
        sim.context.getState(
            getPositions=True,
            enforcePeriodicBox=True).getPositions(),
        file=outfile,
        keepIds=True
    )

# Save and serialize system
logger.info("Serializing system to %s" % oDict['oSystemFile'])
sim.system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
with bz2.open(oDict['oSystemFile'], "wt") as outfile:
    xml = mm.XmlSerializer.serialize(sim.system)
    outfile.write(xml)

# Save and serialize integrator
logger.info("Serializing integrator to %s" % oDict['oIntegratorFile'])
integrator = sim.context.getIntegrator()
with bz2.open(oDict['oIntegratorFile'], "wt") as outfile:
    xml = mm.XmlSerializer.serialize(integrator)
    outfile.write(xml)
