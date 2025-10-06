import pickle

def storePopulation(obj, filename):
    """
    Stores the population object in a file using pickle serialization.

    Parameters:
        obj: Population 
        filename (str): The path or filename where the population should be stored.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Population successfully stored in '{filename}'.")


def loadPopulation(filename):
    """
    Loads a previously Populaiton from a file.

    Parameters:
        filename (str): The path or filename of the stored population.

    Returns:
        The population.
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

