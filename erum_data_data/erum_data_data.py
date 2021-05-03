from .dataset import Dataset
from .load_data import LoadData


class TopTagging(Dataset):
    """
    Top tagging dataset.

    Description:
    14 TeV, hadronic tops for signal, QCD djets background, Delphes ATLAS detector card with Pythia. No MPI/pile-up included
    Particle-flow entries (produced by Delphes E-flow) have been clustered into anti-kT 0.8 jets in the pT range [550,650].
    All top jets are matched to a parton-level top within ∆R = 0.8, and to all top decay partons within 0.8. Also,|eta|_jet < 2 has been required.

    Ref:
    Deep-learned Top Tagging with a Lorentz Layer by A Butter, G Kasieczka, T and M Russell (arXiv: 1707.08966)

    Dataset shape:
    ~2M events have been stored divided between training (~1.6M) and test (~400k)) and the shape of the dataset is (# of events, 200, 4).
    The feature represent the leading 200 jet constituent four-momenta, with zero-padding for jets that have less than 200.
    Constituents are sorted by pT, with the highest pT one first.

    The second dataset that is included is just a flag "ttv" to identify what the event was before the reshaping operated by us. Here a legenda:
        0 = training event;
        1 = test event;
        2 = validation event;

    Note that in the current splitting of the dataset, training and validation events have been merged together as a unique training dataset. So for most intents and purposes one should just train the model on the first dataset and omit the second 'ttv' dataset altogether.

    The set label are 0 for QCD and 1 for top.
    """

    name = "TopTagging"
    filename = "1_top_tagging_2M.npz"
    url = "https://desycloud.desy.de/index.php/s/aZqyNSg4B7nn8qQ/download"
    md5 = "708a8369d75ceff2229bd8c46b47afea"
    task = 'classification'
    load_data = LoadData.TopTagging_data


class Spinodal(Dataset):
    """
    Spinodal dataset.

    Description:
    The classification goal for this dataset is to identify the nature of the QCD phase transitic collisions at the CBM experiment and, in particular, whether signals for b-associated with the phase transition can be ound in the final momentum spectra of certain collisions.

    Ref:
    J. Steinheimer, L. Pang, K. Shou, V. Koch, J. Randrup and H.Stoecker, JHEP 19 doi:10.1007/JHEP12(2019)122 [arXiv:1906.06562 [nucl-th]]

    Dataset shape:
    The dataset is composed of 29'000 2D histograms of shape 20x20 describing pion momenta, divided in training (70%) and test (30%). So, the shape of the dataset is (# of events, 20, 20).

    The set label is 1 for a Spinodal event and 0 for a Maxwell event.
    """

    name = "Spinodal"
    filename = "2_spinodal_29k.npz"
    url = "https://desycloud.desy.de/index.php/s/zZCCSfwwEkT5Pgk/download"
    md5 = "c22326822d9bac7c074467ccccc6fe4f"
    task = 'classification'
    load_data = LoadData.spinodal_data


class EOSL(Dataset):
    """
    EOSL or EOSQ dataset.

    Description:
    The task here is to classify the QCD transition nature (two different equation of state: cross-over EOSL or 1st order EOSQ) happened in heavy-ion collisions from the final state pion spectra. The pion spectra in transverse momentum and azimuthal angle are simulated here with Hybrid (Viscous Hydrodynamics plus hadronic cascade UrQMD) modeling for heavy-ion collisions by varing different physical  arameters (collision energy, entrality, initial time, initial ondition models with fluctuations, shear viscousity, freeze-out temperature, switch time from hydro to hadronic cascade)

    Ref:
    An equation-of-state-meter of quantum chromodynamics transition from deep learning, Long-Gang Pang, Kai Zhou, Nan Su, Hannah Petersen, Horst Stoecker and Xin-Nian Wang, Nature Commun.9 (2018) no.1,210
    Identifying the nature of the QCD transition in relativistic collision of heavy nuclei with deep learning, Yi-Lun Du, Kai Zhou, Jan Steinheimer, Long-Gang Pang, Anton Motornenko, Hong-Shi Zong, Xin-Nian Wang and Horst Stoecker, arXiv:1910.11530

    Dataset shape:
    The dataset is composed of ~180'000 2D histogram of shape 24x24 of the pion spectra and it is divided in 70% training and 30% test. So, the shape of the dataset is (# of events, 24, 24).
    Label is 1 for EOSQ and 0 for EOSL.
    """

    name = "EOSL"
    filename = "3_EOSL_or_EOSQ_180k.npz"
    url = "https://desycloud.desy.de/index.php/s/DT7sWm6rNR5zss9/download"
    md5 = "c070a9743163c3f467ceb87ac4e19fd1"
    task = 'classification'
    load_data = LoadData.eosl_data


class Airshower(Dataset):
    """
    Airshower Regression task: Shower maximum

    Based on https://doi.org/10.1016/j.astropartphys.2017.10.006

    Produced by jonas.glombitza@rwth-aachen.de

    ----------------------------------
    Dataset shape:

    Three sets of input data:
    - first set of input data (shape: [70k, 9, 9, 80]):
        - 70k events (airshowers)
        - 9x9 = 81 ground detector stations
        - 80 measured signal bins (forming one signal trace per station)
        -padding: (-1) padding for instances that the detector / or timestep did not detect a particle

    - second set of input data (shape: [70k, 9, 9, 1]:
        - 70k events (airshowers)
        - 9x9 = 81 ground detector stations
        - 1 starting time of the signal trace (arrival time of first particles at each station)
        - padding: (-1) padding for instances that the detector / or timestep did not detect a particle

    - third set of input data
        - detector geometry - for reference if needed
        - 81 ground detector stations
        - 3 features: x,y,z location of each station

    ----------------------------------
    Label:
    "Xmax" = shower maximum
    For a regression task.
    """

    name = "Airshower"
    filename = "4_airshower_100k_regression.npz"
    url = "https://desycloud.desy.de/index.php/s/YHa79Gx94CbPx8Q/download"
    md5 = "367dc93bec6111a1990f85cc8ff58d1f"
    task = 'regression'
    load_data = LoadData.airshower_data

class Belle(Dataset):
    """
    SmartBKG dataset (Belle II - generated events passing downstream selection)

    The goal of this classification problem is to identify generated events that pass a selection already before the expensive detector simulation and reconstruction.

    Original dataset and additional information: https://github.com/kahn-jms/belle-selective-mc-dataset

    ----------------------------------
    Dataset shape:

    Two sets of input data:
    - first set with shape:
        - 280000 belle collider events
        - 100 particles (zero padded)
        - 9 features ('prodTime', 'energy', 'x', 'y', 'z', 'px', 'py', 'pz', 'PID')
            - note: PID corresponding to a unique PDG particle ID, but mapped to a continous space

    - second set with shape:
        - 280000 belle collider events
        - 100 indices of mother particles (adjacency matrix for creating a graph of the event)
            - note: these are -1 padded
    ----------------------------------
    Label:
    event passes (1) or fails (0) a selection that was applied after detector simulation and reconstruction
    """

    name = "Belle"
    filename = "6_belle_selective_400k.npz"
    url = "https://desycloud.desy.de/index.php/s/RKB4z3mMcPPY982/download"
    md5 = "85b10c9df3903455ab247a0ab4b51e5f"
    task = 'classification'
    load_data = LoadGraph.belle_data
    load_graph_tf_data = lambda *args, **kwargs: LoadGraph.belle_graph(*args, **kwargs, as_tf_data=True)
