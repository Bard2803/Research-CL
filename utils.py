from torch.utils.data import Subset
import numpy as np

def load_balanced_subset_of_experience(experience, fraction=0.5):
    """ Load a balanced subset of the experience """
    classes = []
    for sample_class in experience.dataset:
        sample_class = sample_class[1]
        classes.append(sample_class)
    classes = np.array(classes)
    sample_class, classes_occurance = np.unique(classes, return_counts=True)
    counter = 0
    indices_to_take = []
    for num_occurances in classes_occurance:
        indices = num_occurances*fraction
        indices = range(counter, indices+counter)
        counter += num_occurances
        indices_to_take += indices

    return experience.dataset.subset()