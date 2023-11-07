from constants import *
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def infer_sentences(model, sentences, start, gram, algorithm):
    """
    Args:
        model (POSTagger): model used for inference
        sentences (list[str]): list of sentences to infer by single process
        start (int): index of first sentence in sentences in the original list of sentences

    Returns:
        dict: index, predicted tags for each sentence in sentences
    """
    res = {}
    for i in range(len(sentences)):
        res[start + i] = model.inference(sentences[i], gram, algorithm)
    return res

    
def compute_prob(model, sentences, tags, start, grams):
    """

    Args:
        model (POSTagger): model used for inference
        sentences (list[str]): list of sentences 
        sentences (list[str]): list of tags
        start (int): index of first sentence in sentences in the original list of sentences


    Returns:
        dict: index, probability for each sentence,tag pair
    """
    res = {}
    for i in range(len(sentences)):
        res[start+i] = model.sequence_probability(sentences[i], tags[i], grams)
    return res


#from https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list    
def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    df_sentences = pd.read_csv(open(sentence_file))
    doc_start_indexes = df_sentences.index[df_sentences['word'] == '-DOCSTART-'].tolist()
    num_sentences = len(doc_start_indexes)

    sentences = [] # each sentence is a list of tuples (index,word)
    if tag_file:
        df_tags = pd.read_csv(open(tag_file))
        tags = []
    for i in tqdm(range(num_sentences)):
        index = doc_start_indexes[i]
        if i == num_sentences-1:
            # handle last sentence
            next_index = len(df_sentences)
        else:
            next_index = doc_start_indexes[i+1]

        sent = []
        tag = []
        for j in range(index, next_index):
            word = df_sentences['word'][j].strip()
            if not CAPITALIZATION or word == '-DOCSTART-':
                word = word.lower()
            sent.append(word)
            if tag_file:
                tag.append((df_tags['tag'][j]))
        if STOP_WORD:
            sent.append('<STOP>')
        sentences.append(sent)
        if tag_file:
            if STOP_WORD:
                tag.append('<STOP>')
            tags.append(tag)

    if tag_file:
        return sentences, tags

    return sentences

def confusion_matrix(tag2idx,idx2tag, pred, gt, fname, name):
    """Saves the confusion matrix

    Args:
        tag2idx (dict): tag to index dictionary
        idx2tag (dict): index to tag dictionary
        pred (list[list[str]]): list of predicted tags
        gt (_type_): _description_
        fname (str): filename to save confusion matrix

    """
    matrix = np.zeros((len(tag2idx), len(tag2idx)))
    flat_pred = []
    flat_y = []
    for p in pred:
        flat_pred.extend(p)
    for true in gt:
        flat_y.extend(true)
    for i in range(len(flat_pred)):
        idx_pred = tag2idx[flat_pred[i]]
        idx_y = tag2idx[flat_y[i]]
        matrix[idx_y][idx_pred] += 1

    # Save the diagonal values and set them to 0
    diag_values = np.diag(matrix)
    np.fill_diagonal(matrix, 0)

    # Filter the matrix based on the threshold
    mask = np.any(matrix > 300, axis=1)
    filtered_matrix = matrix[mask][:, mask]

    # Restore the diagonal values
    np.fill_diagonal(filtered_matrix, diag_values[mask])

    # Adjust the labels for the filtered matrix
    filtered_idx2tag = np.array([idx2tag[i] for i in range(len(tag2idx))])[mask]

    df_cm = pd.DataFrame(filtered_matrix, index=filtered_idx2tag, columns=filtered_idx2tag)

    plt.figure(figsize=(35, 20))

    # Creating a custom colormap
    colors = ["white", "darkred"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_red", colors)

    # Using the custom colormap and modifying annotations to display hyphen for zeros
    annot_array = np.array(df_cm.astype(str))
    annot_array[annot_array == '0.0'] = '-'

    ax = sn.heatmap(df_cm, annot=annot_array, cmap=cmap, fmt='', annot_kws={"size": 50})
    # Adjust colorbar font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=50)

    # Set the title and adjust font sizes
    plt.title(name, fontsize=80)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.xlabel('Predicted', fontsize=70)
    plt.ylabel('True', fontsize=70)

    plt.savefig(fname)