import pickle

def storePopulation(obj, filename):
    """
    Store a population object in a file using pickle serialization.

    Parameters
    ----------
    obj : Population
        The population object to store.
    filename : str
        Path or filename where the population should be saved.

    Returns
    -------
    None
        Prints a confirmation message upon successful storage.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Population successfully stored in '{filename}'.")


def loadPopulation(filename):
    """
    Load a previously stored population object from a file.

    Parameters
    ----------
    filename : str
        Path or filename of the stored population.

    Returns
    -------
    Population
        The deserialized population object.
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
