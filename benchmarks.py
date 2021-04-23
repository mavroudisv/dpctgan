import sdgym 

from sdgym.synthesizers import (
    CLBN,
    CopulaGAN,
    CTGAN,
    dpCTGAN,
    #HMA1,
    Identity,
    Independent,
    MedGAN,
    #PAR,
    PrivBN,
    #SDV,
    TableGAN,
    #TVAE,
    Uniform,
    VEEGAN)

all_synthesizers = [
    #CLBN,
    #CTGAN,
    dpCTGAN,
    #CopulaGAN,
    #HMA1,
    #Identity,
    #Independent,
    #MedGAN,
    #PAR,
    #PrivBN,
    #SDV,
    #TVAE,
    #TableGAN,
    #Uniform,
    #VEEGAN,
]

all_datasets = [
    "adult",
    #"alarm",
    #"asia",  #all features are discrete
    #"census",
    #"child",
    #"covtype",
    #"credit",
    #"grid",
    #"gridr",
    #"insurance",
    #"intrusion",
    #"mnist12",
    #"mnist28",
    #"news",
    #"ring"
]

scores = sdgym.run(synthesizers=all_synthesizers, 
                    datasets=all_datasets,
					show_progress=False,
                    #output_path="/SAN/infosec/TLS_fingerprinting/experiments/SDGym/leaderboard.csv",
                    iterations=1)
print("Scores", scores)