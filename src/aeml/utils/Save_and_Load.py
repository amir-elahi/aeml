import pickle

def save_to_pickle(*pickles, output_file, location='/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Output/'):
    file_path = location + output_file
    with open(file_path, 'wb') as file:
        pickle.dump(pickles, file)

def load_from_pickle(input_file, location='/home/lsmo/Desktop/aeml_project/aeml/DataDynamo/Output/'):
    file_path = location + input_file
    with open(file_path, 'rb') as file:
        pickles = pickle.load(file)
    return pickles