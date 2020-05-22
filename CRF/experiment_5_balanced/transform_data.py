import shutil


def run():
    shutil.copy('Data/experiment_3_strip1/vector_vector_znacilke.pickle',
                'Data/experiment_5_balanced/vector_vector_znacilke.pickle')

    shutil.copy('Data/experiment_3_strip1/vector_vector_classes.pickle',
                'Data/experiment_5_balanced/vector_vector_classes.pickle')
