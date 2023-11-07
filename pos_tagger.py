import string
from multiprocessing import Pool
import time
from utils import *
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
import sys
import argparse

import warnings
# Suppress the runtime warning
warnings.filterwarnings("ignore", category=RuntimeWarning)


def display_misclassifications(sentences, tags, predictions):
    # Assuming n is the length of sentences, tags, and predictions
    n = len(sentences)

    # Step 1: Create a dictionary to count misclassifications for each tag
    misclassified_dict = defaultdict(lambda: {'count': 0, 'samples': []})

    # Step 2: Populate the dictionary with data
    for i in range(n):
        for j in range(len(sentences[i])):
            true_tag = tags[i][j]
            predicted_tag = predictions[i][j]
            # If the tags don't match, it's a misclassification
            if true_tag != predicted_tag:
                misclassified_dict[true_tag]['count'] += 1
                if len(misclassified_dict[true_tag]['samples']) < 10:
                    misclassified_dict[true_tag]['samples'].append((sentences[i][j], predicted_tag))

    # Calculate the total number of misclassified tokens
    total_misclassified_tokens = sum(misclassified_dict[tag]['count'] for tag in misclassified_dict)

    # Step 3: Sort the dictionary based on misclassification count
    sorted_misclassified_tags = sorted(misclassified_dict.keys(), key=lambda x: misclassified_dict[x]['count'],
                                       reverse=True)

    # Step 4: Display top 5 misclassified tags with samples
    for tag in sorted_misclassified_tags[:5]:
        percentage_misclassified = (misclassified_dict[tag]['count'] / total_misclassified_tokens) * 100
        print(f"True Tag: {tag}")
        print(f"Number of Misclassifications: {misclassified_dict[tag]['count']} ({percentage_misclassified:.2f}%)")
        print("Sample Tokens and Their Predicted Tags:")
        for sample in misclassified_dict[tag]['samples']:
            print(f"Token: {sample[0]}, Predicted Tag: {sample[1]}")
        print("------")


class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleDNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, num_classes)

        self.leakyrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(32)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.batchnorm1(x1)
        x1 = self.leakyrelu(x1)
        x1 = self.dropout(x1)

        x2 = self.fc2(x1)
        x2 = self.batchnorm2(x2)
        x2 = self.leakyrelu(x2)
        x2 = self.dropout(x2)

        x3 = self.fc3(x2)
        x3 = self.batchnorm3(x3)
        x3 = self.leakyrelu(x3)
        x3 = self.dropout(x3)

        x4 = self.fc4(x3)
        x4 = self.batchnorm4(x4)
        x4 = self.leakyrelu(x4)
        x4 = self.dropout(x4)

        x5 = self.fc5(x4)
        output = F.softmax(x5, dim=1)

        return output


def evaluate(data, model, grams, algorithm):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    processes = 12
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i, grams, algorithm]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")

    predictions_prob = [predictions[i] for i in range(len(sentences))]
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], predictions_prob[i:i+k], i, grams]))
    ans = [r.get(timeout=None) for r in res]
    probabilities_pred = dict()
    for a in ans:
        probabilities_pred.update(a)
    print(f"Probability Estimation for Predict Sequence Runtime: {(time.time()-start)/60} minutes.")


    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i, grams]))
    ans = [r.get(timeout=None) for r in res]
    probabilities_gold = dict()
    for a in ans:
        probabilities_gold.update(a)
    print(f"Probability Estimation for Gold Sequence Runtime: {(time.time()-start)/60} minutes.")

    # Count of sub-optimal results
    count_sub_optimal = sum(1 for i in range(len(sentences)) if probabilities_pred[i] < probabilities_gold[i])
    # Calculate the percentage
    percentage_sub_optimal = (count_sub_optimal / len(sentences)) * 100
    # Extract 3 sample sentences where probability_pred < probability_gold
    sub_optimal_sentences = [sentences[i] for i in range(len(sentences)) if
                             probabilities_pred[i] < probabilities_gold[i]]
    sample_sentences = sub_optimal_sentences[:3]

    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens

    display_misclassifications(sentences, tags, predictions)


    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))
    print("------------")
    print("Mean Pred Sequence Probabilities/Score: {}".format(sum(probabilities_pred.values())/n))
    print("Mean Gold Sequence Probabilities/Score: {}".format(sum(probabilities_gold.values())/n))
    gold_score = sum(probabilities_gold.values())/n
    mean_prob = sum(probabilities_pred.values())/n
    print("Gold sequence score is", "larger" if gold_score > mean_prob else "smaller",
          "than predicted sequence mean score")
    print("------------")
    print(f"Percentage of sub-optimal results: {percentage_sub_optimal:.2f}%")
    for idx, sentence in enumerate(sample_sentences, 1):
        print(f"Sample {idx}: {' '.join(sentence)}")
    
    confusion_matrix(model.tag2idx, model.idx2tag, predictions.values(), tags,
                     f'image/{grams}_{algorithm}_cm.png',
                     f'{grams} {algorithm} Confusion Matrix')

    #return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


def token_classifier_data(self, train_data):
    sentences, tags = train_data
    # Define a filename
    output_filename = "data/words_and_tags_comb.csv"
    # Open a file in write mode
    with open(output_filename, 'w') as outfile:
        # Write headers to the CSV file
        outfile.write('"Word","True tag"\n')
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                # Write the token and its true tag to the file in CSV format, with fields quoted
                outfile.write(f'"{sentences[i][j]}","{tags[i][j]}"\n')
    print(f"Token and tag results saved to {output_filename}")


def output_test(pos_tagger, gram, algorithm, path):
    print("Testing.....")
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    sentences = train_data[0] + dev_data[0]
    tags = train_data[1] + dev_data[1]
    train_data = [sentences, tags]

    pos_tagger.train(train_data, 'k')
    pos_tagger.train_or_load_UKN_classifier()

    # Predict tags for the test set
    test_predictions = []
    for sentence in tqdm(test_data, desc="Processing sentences"):
        test_predictions.extend(pos_tagger.inference(sentence, gram, algorithm)[:-1])

    # Write them to a file to update the leaderboard
    # TODO

    df = pd.DataFrame({
        'id': range(0, len(test_predictions)),
        'tag': test_predictions
    })

    # Write the DataFrame to a CSV file
    df.to_csv(path, index=False)


def output_dev(pos_tagger, gram, algorithm):
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv")

    pos_tagger.train(train_data, 'k')
    pos_tagger.train_or_load_UKN_classifier()

    # Predict tags for the test set
    dev_predictions = []
    for sentence in tqdm(dev_data, desc="Processing sentences"):
        dev_predictions.extend(pos_tagger.inference(sentence, gram, algorithm)[:-1])

    # Write them to a file to update the leaderboard
    # TODO

    df = pd.DataFrame({
        'id': range(0, len(dev_predictions)),
        'tag': dev_predictions
    })

    # Write the DataFrame to a CSV file
    df.to_csv('data/dev_pred.csv', index=False)

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == -np.inf for a in args):
        return -np.inf
    a_max = np.max(args)
    lsp = np.log(sum(np.exp(a - a_max) for a in args))
    return a_max + lsp


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        self.all_words = None
        self.all_tags = None

        self.emissions = None
        self.fourgrams = None
        self.trigrams = None
        self.bigrams = None
        self.unigrams = None

        self.word2idx = {}
        self.idx2word = {}
        self.tag2idx = {}
        self.idx2tag = {}

        self.clf = None

    def __build_tag_index(self, tags):
        """
        Map each tag to a unique integer.
        """
        self.all_tags = list(set([t for tag in tags for t in tag]))
        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}

    def __build_word_index(self, sentences):
        """
        Map each word to a unique integer.
        """
        word_freq = {}
        for sentence in sentences:
            for word in sentence:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Filter out words that appear fewer than 5 times and replace with 'UKN'
        reduced_words = [word for sentence in sentences for word in sentence]
        # if word_freq[word] >= 2 else 'UKN'

        # Create a set of the words (removes duplicates)
        self.all_words = list(set(reduced_words))

        # Update mappings
        self.word2idx = {self.all_words[i]: i for i in range(len(self.all_words))}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __get_unigrams(self, tags):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        ## TODO
        for sentence in tags:
            for tag in sentence:
                self.unigrams[self.tag2idx[tag]] += 1

        # Add-one smoothing
        self.unigrams += 1
        # Convert counts to probabilities
        self.unigrams /= (self.unigrams.sum() + len(self.all_tags) * 1)

    def __get_bigrams(self, tags):
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 
        """
        ## TODO
        for sentence in tags:
            for i in range(1, len(sentence)):
                self.bigrams[self.tag2idx[sentence[i - 1]], self.tag2idx[sentence[i]]] += 1

        # Add-one smoothing
        self.bigrams += 1
        # Convert counts to probabilities
        total_bigrams = self.bigrams.sum(axis=1, keepdims=True) + len(self.all_tags)
        self.bigrams /= total_bigrams

    def __get_trigrams(self, tags, smooth_method):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        ## TODO
        if smooth_method == 'k':
            self.__get_trigram_k_smoothing(tags)
        elif smooth_method == "no":
            self.__get_trigram_no_smoothing(tags)
        else:
            self.__get_trigrams_linear_interpolation(tags)

    def __get_trigram_no_smoothing(self, tags):
        for sentence in tags:
            for i in range(2, len(sentence)):
                self.trigrams[self.tag2idx[sentence[i - 2]], self.tag2idx[sentence[i - 1]], self.tag2idx[sentence[i]]] += 1

        self.trigrams += 1e-10
        # Convert counts to probabilities
        total_trigrams = self.trigrams.sum(axis=(1, 2), keepdims=True)
        self.trigrams /= total_trigrams

    def __get_trigram_k_smoothing(self, tags):
        for sentence in tags:
            for i in range(2, len(sentence)):
                self.trigrams[self.tag2idx[sentence[i - 2]], self.tag2idx[sentence[i - 1]], self.tag2idx[sentence[i]]] += 1

        # Add-one smoothing
        self.trigrams += 1
        # Convert counts to probabilities
        total_trigrams = self.trigrams.sum(axis=(1, 2), keepdims=True) #+ len(self.all_tags) ** 2
        self.trigrams /= total_trigrams

    def __get_trigrams_linear_interpolation(self, tags, lambda1=0.5, lambda2=0.3, lambda3=0.2):
        """
        Computes trigrams using linear interpolation for smoothing with fixes for division by zero.
        """

        # Initialize trigram counts to zero
        num_tags = len(self.tag2idx)
        self.trigrams += 1  # Using float64
        bigrams = np.zeros((num_tags, num_tags), dtype=np.float64)
        unigrams = np.zeros(num_tags, dtype=np.float64)

        # Compute trigram frequencies, bigram frequencies, and unigram counts
        for sentence in tags:
            for i, tag in enumerate(sentence):
                unigrams[self.tag2idx[tag]] += 1
                if i > 0:
                    bigrams[self.tag2idx[sentence[i - 1]], self.tag2idx[tag]] += 1
                if i > 1:
                    self.trigrams[self.tag2idx[sentence[i - 2]], self.tag2idx[sentence[i - 1]], self.tag2idx[tag]] += 1

        # Convert counts to probabilities with checks for division by zero
        total_tags = unigrams.sum()
        unigram_probs = unigrams / (total_tags + 1e-10)

        total_bigrams = bigrams.sum(axis=1, keepdims=True)
        bigram_probs = np.divide(bigrams, total_bigrams + 1e-10, where=total_bigrams != 0)

        total_trigrams = self.trigrams.sum(axis=(1, 2), keepdims=True)
        trigram_probs = np.divide(self.trigrams, total_trigrams + 1e-10, where=total_trigrams != 0)

        # Apply linear interpolation
        for tag_i in range(num_tags):
            for tag_i_minus_1 in range(num_tags):
                for tag_i_minus_2 in range(num_tags):
                    self.trigrams[tag_i_minus_2, tag_i_minus_1, tag_i] = (
                            lambda1 * trigram_probs[tag_i_minus_2, tag_i_minus_1, tag_i] +
                            lambda2 * bigram_probs[tag_i_minus_2, tag_i_minus_1] +
                            lambda3 * unigram_probs[tag_i]
                    )

        return self.trigrams

    def __get_fourgram(self, tags, k=1):
        for sentence in tags:
            for i in range(3, len(sentence)):
                self.fourgrams[
                    self.tag2idx[sentence[i - 3]],
                    self.tag2idx[sentence[i - 2]],
                    self.tag2idx[sentence[i - 1]],
                    self.tag2idx[sentence[i]]
                ] += 1

        # k-smoothing
        self.fourgrams += k
        # Convert counts to probabilities
        total_fourgrams = self.fourgrams.sum(axis=(1, 2, 3), keepdims=True)  # + len(self.all_tags) ** 3
        self.fourgrams /= total_fourgrams

    def __get_emissions(self, sentences, tags, smooth_method):
        """
        Computes emission probabilities.
        Tip. Map each tag to an integer and each word in the vocabulary to an integer.
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag)
        """
        ## TODO
        if smooth_method == 'k':
            self.__get_emission_k_smoothing(sentences, tags)
        elif smooth_method == "no":
            self.__get_emission_no_smoothing(sentences, tags)
        else:
            self.__get_emissions_linear_interpolation(sentences, tags)

    def __get_emission_no_smoothing(self, sentences, tags):
        for s, t in zip(sentences, tags):
            for word, tag in zip(s, t):
                # Check if word exists in word2idx else use 'ukn'
                word_idx = self.word2idx[word]#.get(word, self.word2idx['UKN'])
                self.emissions[self.tag2idx[tag], word_idx] += 1

        # Convert counts to probabilities
        total_emissions = self.emissions.sum(axis=1, keepdims=True)
        self.emissions = self.emissions / total_emissions

    def __get_emission_k_smoothing(self, sentences, tags):
        for s, t in zip(sentences, tags):
            for word, tag in zip(s, t):
                # Check if word exists in word2idx else use 'ukn'
                word_idx = self.word2idx[word]#.get(word, self.word2idx['UKN'])
                self.emissions[self.tag2idx[tag], word_idx] += 1

        # Add-one smoothing
        self.emissions += 1e-10

        # Convert counts to probabilities
        total_emissions = self.emissions.sum(axis=1, keepdims=True)
        self.emissions = self.emissions / total_emissions

    def __get_emissions_linear_interpolation(self, sentences, tags, lambda_coeff=0.7):
        """
        Computes emission probabilities using linear interpolation for smoothing.
        """

        # Initialize emission counts to zero
        num_tags = len(self.tag2idx)
        num_words = len(self.word2idx)
        self.emissions = np.zeros((num_tags, num_words), dtype=int)
        word_counts = np.zeros(num_words, dtype=int)

        # Compute emission frequencies and word counts
        for s, t in zip(sentences, tags):
            for word, tag in zip(s, t):
                word_idx = self.word2idx.get(word)
                self.emissions[self.tag2idx[tag], word_idx] += 1
                word_counts[word_idx] += 1

        # Convert counts to probabilities
        total_emissions = self.emissions.sum(axis=1, keepdims=True)
        self.emissions = self.emissions / total_emissions

        # Compute word probabilities P(w)
        total_words = word_counts.sum()
        word_probs = word_counts / total_words

        # Apply linear interpolation
        for tag_idx in range(num_tags):
            for word_idx in range(num_words):
                self.emissions[tag_idx, word_idx] = (lambda_coeff * self.emissions[tag_idx, word_idx] +
                                                     (1 - lambda_coeff) * word_probs[word_idx])

        return self.emissions

    def __extract_UKN_features(self, word, prev_tag, pprev_tag): #prev_tag=None, prev_prev_tag=None, next_tag=None, next_next_tag=None
        word_lower = word.lower()

        # Define a function to convert word to its shape
        def word_shape(w):
            shape = []
            for char in w:
                if char.isalpha():
                    if char.isupper():
                        shape.append('A')
                    else:
                        shape.append('a')
                elif char.isdigit():
                    shape.append('9')
                else:
                    shape.append(char)
            return ''.join(shape)

        features = {
            'word_lower': word_lower,
            'suffix-1': word_lower[-1:],
            'suffix-2': word_lower[-2:],
            'suffix-3': word_lower[-3:],
            'prefix-1': word_lower[:1],
            'prefix-2': word_lower[:2],
            'prefix-3': word_lower[:3],
            'is_capitalized': word[0].upper() == word[0],
            'has_hyphen': '-' in word,
            'is_numeric': word.isdigit(),
            'length': len(word),
            'word_shape': word_shape(word),
            'vowel_consonant_pattern': ''.join(
                ['v' if char in 'aeiou' else 'c' for char in word_lower if char.isalpha()]),
            'has_digit': any(char.isdigit() for char in word),
            'has_uppercase': any(char.isupper() for char in word),
            'has_punctuation': any(char in string.punctuation for char in word),
            'vowel_count': sum(1 for char in word_lower if char in 'aeiou'),
            'consonant_count': sum(1 for char in word_lower if char.isalpha() and char not in 'aeiou'),
            'uppercase_count': sum(1 for char in word if char.isupper()),
            'is_title': word.istitle(),
            'ends_in_ing': word_lower.endswith('ing'),
            'ends_in_ed': word_lower.endswith('ed'),
            'ends_in_ly': word_lower.endswith('ly'),
            # ... add more features here as needed
            'hyphen_segments': word.count('-') + 1 if '-' in word else 0,
            'is_currency': word in ["$", "£", "€", "¥"],
            'is_special_char': word in string.punctuation,
            #'prev_word': prev_tag.lower() if prev_tag else 'NONE',
            #'pprev_word': pprev_tag.lower() if pprev_tag else 'NONE',
        }
        return features

    def train_or_load_UKN_classifier(self):
        # Check if a pre-trained model exists
        model_path = "model/simplednn_model_4.pth"
        if os.path.exists(model_path):
            df = pd.read_csv("data/words_and_tags_comb.csv")
            train_words = df["Word"].tolist()
            train_tags = df["True tag"].tolist()

            X_train = [self.__extract_UKN_features(train_words[i],
                                                   train_tags[i - 1] if i - 1 >= 0 else None,
                                                 '^' if (i - 1 >= 0 and train_tags[i - 1] == 'O') else (
                                                     train_tags[i - 2] if i - 2 >= 0 else None),
                                                   ) for i in range(len(train_words))]

            vectorizer = DictVectorizer(sparse=False)
            X_train_vec = vectorizer.fit_transform(X_train)
            encoder = LabelEncoder()
            encoder.fit_transform(train_tags)
            device = torch.device("cpu")

            self.clf = SimpleDNN(input_dim=X_train_vec.shape[1], num_classes=len(encoder.classes_)).to(device)
            self.clf.load_state_dict(torch.load(model_path))
            self.clf.eval()  # Set the model to evaluation mode
            self.vectorizer = vectorizer
            self.encoder = encoder

        else:
            df = pd.read_csv("data/words_and_tags_comb.csv")
            train_words = df["Word"].tolist()
            train_tags = df["True tag"].tolist()

            X_train = [self.__extract_UKN_features(train_words[i],
                                                   train_tags[i - 1] if i - 1 >= 0 else None,
                                                 '^' if (i - 1 >= 0 and train_tags[i - 1] == 'O') else (
                                                     train_tags[i - 2] if i - 2 >= 0 else None),
                                                   ) for i in range(len(train_words))]

            vectorizer = DictVectorizer(sparse=False)
            X_train_vec = vectorizer.fit_transform(X_train)
            encoder = LabelEncoder()
            encoded_Y = encoder.fit_transform(train_tags)
            #encoder.fit(train_tags)

            # Assuming you want to use the 'mps' device
            device = torch.device("cpu")

            # Convert the data to tensors
            X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(encoded_Y, dtype=torch.long).to(device)

            # Create a dataset and dataloader for batching
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            dataloader = DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True)

            # Initialize the model, criterion, and optimizer
            model = SimpleDNN(input_dim=X_train_vec.shape[1], num_classes=len(encoder.classes_)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            # Training loop
            for epoch in range(10):
                # Create a tqdm object around the dataloader to display the progress bar
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/10", leave=True, position=0)
                for inputs, labels in progress_bar:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Update the progress bar description with the current loss
                    progress_bar.set_description(f"Epoch {epoch + 1}/10 - Loss: {loss.item():.4f}")

            # Save the trained model, vectorizer, and label encoder
            torch.save(model.state_dict(), model_path)
            self.clf = model
            self.vectorizer = vectorizer
            self.encoder = encoder

    def __determine_tag_ml_prob(self, word, prev_tag, pprev_tag):

        features = self.__extract_UKN_features(word, prev_tag, pprev_tag)
        features_vec = self.vectorizer.transform([features])
        inputs = torch.tensor(features_vec, dtype=torch.float32).to(torch.device("cpu"))

        # Ensure model is in evaluation mode
        self.clf.eval()

        with torch.no_grad():
            outputs = self.clf(inputs)
            probabilities = nn.Softmax(dim=1)(outputs)
            predicted_index = torch.argmax(probabilities).item()
            max_probability = probabilities[0][predicted_index].item()
            predicted_tag = self.encoder.inverse_transform([predicted_index])[0]

        return predicted_tag

    def __handle_unknown_word(self, word, prev_tag, pprev_tag):
        """Handles unknown words by determining their tag and updating the word2idx and emissions matrix."""
        word_idx = self.word2idx.get(word, None)
        if word_idx is None:
            determined_tag = self.__determine_tag_ml_prob(word, prev_tag, pprev_tag)
            if determined_tag:
                self.word2idx[word] = len(self.word2idx)
                word_idx = self.word2idx[word]
                self.emissions = np.hstack((self.emissions, np.zeros((self.emissions.shape[0], 1))))
                self.emissions[self.tag2idx[determined_tag], word_idx] = 1
        return word_idx

    def train(self, data, smooth_method='k'):
        """Trains the model by computing transition and emission probabilities.
        """
        sentences, tags = data

        # Pad sentences and tags with start token for trigram modeling
        for i in range(len(sentences)):
            sentences[i] = ['^'] + sentences[i]
            tags[i] = ['^'] + tags[i]

        # Build the word and tag indices
        self.__build_word_index(sentences)
        self.__build_tag_index(tags)

        # Now initialize the arrays based on the sizes of tag2idx and word2idx
        tag_count = len(self.tag2idx)
        word_count = len(self.word2idx)
        self.unigrams = np.zeros(tag_count)
        self.bigrams = np.zeros((tag_count, tag_count))
        self.trigrams = np.zeros((tag_count, tag_count, tag_count))
        self.fourgrams = np.zeros((tag_count, tag_count, tag_count, tag_count))
        self.emissions = np.zeros((tag_count, word_count))

        # Calculate unigrams, bigrams, trigrams, and emissions
        self.__get_unigrams(tags)
        self.__get_bigrams(tags)
        self.__get_trigrams(tags, smooth_method)
        self.__get_fourgram(tags)
        self.__get_emissions(sentences, tags, smooth_method)

    def sequence_probability(self, sequence, tags, grams=3):
        if grams == 1:
            return self.__sequence_probability_unigrams(sequence, tags)
        elif grams == 2:
            return self.__sequence_probability_bigrams(sequence, tags)
        elif grams == 3:
            return self.__sequence_probability_trigrams(sequence, tags)
        else:
            return self.__sequence_probability_fourgrams(sequence, tags)

    def __sequence_probability_unigrams(self, sequence, tags):
        """Computes the probability of a tagged sequence given only the emission probabilities."""

        # Ensure the sequence and tags start with 'O'
        if sequence[0] != '-docstart-' or tags[0] != 'O':
            raise ValueError("Both sequence and tags should start with 'O'")

        log_prob = 0.0

        for i in range(len(sequence)):
            word_idx = self.__handle_unknown_word(sequence[i], tags[i - 1], tags[i - 2])
            tag_idx = self.tag2idx[tags[i]]

            # Only Emission Log Probability is considered for unigrams
            emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

            log_prob += emission_log_prob

        return log_prob

    def __sequence_probability_bigrams(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission and bigram transition probabilities."""

        # Ensure the sequence and tags start with 'O'
        if sequence[0] != '-docstart-' or tags[0] != 'O':
            raise ValueError("Both sequence and tags should start with 'O'")

        log_prob = 0.0

        for i in range(1, len(sequence)):
            word_idx = self.__handle_unknown_word(sequence[i], tags[i - 1], tags[i - 2])
            tag_idx = self.tag2idx[tags[i]]

            # Emission Log Probability
            emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)
            log_prob += emission_log_prob

            # Transition Log Probability for Bigrams
            prev_tag_idx = self.tag2idx[tags[i - 1]]
            transition_log_prob = np.log(self.bigrams[prev_tag_idx, tag_idx] + 1e-10)
            log_prob += transition_log_prob

        return log_prob

    def __sequence_probability_trigrams(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition probabilities."""

        # Ensure the sequence and tags start with 'O'
        if sequence[0] != '-docstart-' or tags[0] != 'O':
            raise ValueError("Both sequence and tags should start with 'O'")

        # Pad the sequence and tags for trigram transitions
        sequence = ['^'] + sequence
        tags = ['^'] + tags

        log_prob = 0.0

        for i in range(2, len(sequence)):
            word_idx = self.__handle_unknown_word(sequence[i], tags[i - 1], tags[i - 2])
            tag_idx = self.tag2idx.get(tags[i], None)

            # Emission Log Probability
            emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

            # Transition Log Probability
            prev_tag_idx1 = self.tag2idx[tags[i - 2]]
            prev_tag_idx2 = self.tag2idx[tags[i - 1]]
            transition_log_prob = np.log(self.trigrams[prev_tag_idx1, prev_tag_idx2, tag_idx] + 1e-10)

            log_prob += (emission_log_prob + transition_log_prob)

        return log_prob

    def __sequence_probability_fourgrams(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition probabilities."""

        # Ensure the sequence and tags start with 'O'
        if sequence[0] != '-docstart-' or tags[0] != 'O':
            raise ValueError("Both sequence and tags should start with 'O'")

        # Pad the sequence and tags for four-gram transitions
        sequence = ['^', '^'] + sequence
        tags = ['^', '^'] + tags

        log_prob = 0.0

        for i in range(3, len(sequence)):
            word_idx = self.__handle_unknown_word(sequence[i], tags[i - 1], tags[i - 2])
            tag_idx = self.tag2idx[tags[i]]

            # Emission Log Probability
            emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

            # Transition Log Probability for Four-grams
            prev_tag_idx1 = self.tag2idx[tags[i - 3]]
            prev_tag_idx2 = self.tag2idx[tags[i - 2]]
            prev_tag_idx3 = self.tag2idx[tags[i - 1]]
            transition_log_prob = np.log(self.fourgrams[prev_tag_idx1, prev_tag_idx2, prev_tag_idx3, tag_idx] + 1e-10)

            log_prob += (emission_log_prob + transition_log_prob)

        return log_prob

    def inference(self, sequence, gram=3, algorithm='beam search'):
        """Tags a sequence with part of speech tags.

        """
        ## TODO
        """Tags a sequence with part of speech tags."""
        if gram == 1 and algorithm == 'greedy search':
            return self.__greedy_search_unigrams(sequence)
        elif gram == 1 and algorithm == 'beam search':
            return self.__beam_search_unigrams(sequence)
        elif gram == 1 and algorithm == 'viterbi':
            return self.__viterbi_unigrams(sequence)
        elif gram == 2 and algorithm == 'greedy search':
            return self.__greedy_search_bigrams(sequence)
        elif gram == 2 and algorithm == 'beam search':
            return self.__beam_search_bigrams(sequence)
        elif gram == 2 and algorithm == 'viterbi':
            return self.__viterbi_bigrams(sequence)
        elif gram == 3 and algorithm == 'greedy search':
            return self.__greedy_search_trigrams(sequence)
        elif gram == 3 and algorithm == 'beam search':
            return self.__beam_search_trigrams(sequence)
        elif gram == 3 and algorithm == 'viterbi':
            return self.__viterbi_trigrams(sequence)
        elif gram == 4 and algorithm == 'greedy search':
            return self.__greedy_search_fourgrams(sequence)
        elif gram == 4 and algorithm == 'beam search':
            return self.__beam_search_fourgrams(sequence)
        elif gram == 4 and algorithm == 'viterbi':
            return self.__viterbi_fourgrams(sequence)

    def __viterbi_unigrams(self, sequence):
        n = len(sequence)
        num_tags = len(self.tag2idx)

        # Initialization
        dp = np.full((n, num_tags), -np.inf)
        dp[0][self.tag2idx['O']] = 0  # Starting symbol

        # Recursion
        for i in range(1, n):
            for curr_tag_idx in range(num_tags):
                word_idx = self.__handle_unknown_word(sequence[i], None, None)
                emission_val = self.emissions[curr_tag_idx, word_idx] + 1e-10
                score = np.log(emission_val)

                # Check if this score is better than the previous maximum for the current tag
                if score > dp[i][curr_tag_idx]:
                    dp[i][curr_tag_idx] = score

        # Backtracking to find the best sequence
        best_tags = []
        for i in range(n):
            best_tag_idx = np.argmax(dp[i])
            best_tags.append(self.idx2tag[best_tag_idx])

        best_tags[-1] = '<STOP>'
        return best_tags

    def __viterbi_bigrams(self, sequence):
        n = len(sequence)
        num_tags = len(self.tag2idx)

        # Initialization
        dp = np.full((n, num_tags), -np.inf)
        backpointer = np.zeros((n, num_tags), dtype=int)
        dp[0][self.tag2idx['O']] = 0  # Starting symbol

        # Recursion
        for i in range(1, n):
            for curr_tag_idx in range(num_tags):
                word_idx = self.__handle_unknown_word(sequence[i], None, None)
                emission_val = np.log(self.emissions[curr_tag_idx, word_idx] + 1e-10)

                for prev_tag_idx in range(num_tags):
                    transition_val = np.log(self.bigrams[prev_tag_idx, curr_tag_idx] + 1e-10)
                    score = dp[i - 1][prev_tag_idx] + emission_val + transition_val
                    if score > dp[i][curr_tag_idx]:
                        dp[i][curr_tag_idx] = score
                        backpointer[i][curr_tag_idx] = prev_tag_idx

        # Backtracking
        best_tags = [np.argmax(dp[-1])]
        for i in range(n - 1, 0, -1):
            best_tags.append(backpointer[i][best_tags[-1]])
        best_tags = best_tags[::-1]
        best_tags = [self.idx2tag[idx] for idx in best_tags]
        best_tags[-1] = '<STOP>'
        return best_tags

    def __viterbi_trigrams(self, sequence):
        sequence = ['^'] + sequence  # Pad the sequence with '^'
        # Pre-compute word indices for the sequence
        word_indices = [self.__handle_unknown_word(word, None, None) for word in sequence]

        n = len(sequence)
        num_tags = len(self.tag2idx)

        epsilon = 1e-10
        self.emissions[self.emissions <= 0] = epsilon
        self.trigrams[self.trigrams <= 0] = epsilon
        self.log_emissions = np.log(self.emissions)
        self.log_trigrams = np.log(self.trigrams)

        # Step 1: Initialization
        dp = -np.inf * np.ones((n, num_tags))
        backpointer = []

        # Base case
        dp[0][self.tag2idx['^']] = 0  # Starting symbol
        dp[1][self.tag2idx['O']] = 0  # Assuming 'O' after '^'

        # Step 2: Recursion
        for i in range(2, n):
            backpointer_row = []
            for curr_tag_idx in range(num_tags):
                emission_log_prob = self.log_emissions[curr_tag_idx, word_indices[i]]
                transition_log_probs = self.log_trigrams[:, :, curr_tag_idx]
                total_scores = dp[i - 1, np.newaxis] + emission_log_prob + transition_log_probs
                max_score_prev_tag = np.max(total_scores, axis=1)
                best_prev_tag = np.argmax(total_scores, axis=1)

                dp[i][curr_tag_idx] = np.max(max_score_prev_tag)
                backpointer_row.append(best_prev_tag[np.argmax(max_score_prev_tag)])

            backpointer.append(backpointer_row)

        # Step 3: Termination and backtracking
        best_tags = [np.argmax(dp[-1])]
        for i in range(n - 1, 1, -1):
            best_tags.append(backpointer[i - 2][best_tags[-1]])
        best_tags = best_tags[::-1]
        return [self.idx2tag[idx] for idx in best_tags]

    def __viterbi_fourgrams(self, sequence):
        sequence = ['^', '^'] + sequence  # Pad the sequence with two '^'
        word_indices = [self.__handle_unknown_word(word, None, None) for word in sequence]

        n = len(sequence)
        num_tags = len(self.tag2idx)

        epsilon = 1e-10
        self.emissions[self.emissions <= 0] = epsilon
        self.fourgrams[self.fourgrams <= 0] = epsilon
        self.log_emissions = np.log(self.emissions)
        self.log_fourgrams = np.log(self.fourgrams)

        dp = -np.inf * np.ones((n, num_tags), dtype=np.float64)
        backpointer = []

        dp[0][self.tag2idx['^']] = 0
        dp[1][self.tag2idx['^']] = 0
        dp[2][self.tag2idx['O']] = 0

        for i in range(3, n):
            backpointer_row = []
            for curr_tag_idx in range(num_tags):
                emission_log_prob = self.log_emissions[curr_tag_idx, word_indices[i]]
                transition_log_probs = self.log_fourgrams[:, :, :, curr_tag_idx]

                total_scores = (dp[i - 3, :, np.newaxis, np.newaxis] +
                                dp[i - 2, np.newaxis, :, np.newaxis] +
                                dp[i - 1, np.newaxis, np.newaxis, :] +
                                emission_log_prob +
                                transition_log_probs)

                # Using logsumexp for stability
                total_scores_stable = np.array(
                    [[logsumexp(*total_scores[x, y, :]) for y in range(num_tags)] for x in range(num_tags)])

                max_score_indices = np.unravel_index(np.argmax(total_scores_stable, axis=None),
                                                     total_scores_stable.shape)
                dp[i][curr_tag_idx] = total_scores_stable[max_score_indices]
                backpointer_row.append(max_score_indices[1])  # Storing only the immediate previous tag

            backpointer.append(backpointer_row)

        best_tags = [np.argmax(dp[-1])]
        for i in range(n - 1, 2, -1):
            best_prev1_tag = backpointer[i - 3][best_tags[-1]]
            best_tags.append(best_prev1_tag)
        best_tags = best_tags[::-1]
        print([self.idx2tag[idx] for idx in best_tags])
        return [self.idx2tag[idx] for idx in best_tags]

    def __beam_search_unigrams(self, sequence):
        k = 20  # Number of candidates to keep at each step

        # Initialize k candidate sequences with 'O'
        candidates = [{'tag': ['O'], 'log_prob': 0.0}]

        for i in range(1, len(sequence)):
            all_next_candidates = []

            for curr_candidate in candidates:
                curr_tag = curr_candidate['tag']
                curr_log_prob = curr_candidate['log_prob']

                for possible_tag in self.tag2idx.keys():
                    new_tag_sequence = curr_tag + [possible_tag]

                    word_idx = self.__handle_unknown_word(sequence[i], None, None)
                    tag_idx = self.tag2idx[possible_tag]

                    # Emission Log Probability
                    emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

                    new_log_prob = curr_log_prob + emission_log_prob

                    all_next_candidates.append({'tag': new_tag_sequence, 'log_prob': new_log_prob})

            # Retain only the top k of these candidates
            all_next_candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            candidates = all_next_candidates[:k]

        best_tags = candidates[0]['tag']
        # Return the best tags
        return best_tags

    def __beam_search_bigrams(self, sequence):
        k = 20  # Number of candidates to keep at each step

        # Initialize k candidate sequences with 'O'
        candidates = [{'tag': ['O'], 'log_prob': 0.0}]

        for i in range(1, len(sequence)):
            all_next_candidates = []

            for curr_candidate in candidates:
                curr_tag = curr_candidate['tag']
                curr_log_prob = curr_candidate['log_prob']

                for possible_tag in self.tag2idx.keys():
                    new_tag_sequence = curr_tag + [possible_tag]

                    word_idx = self.__handle_unknown_word(sequence[i], None, None)
                    tag_idx = self.tag2idx[possible_tag]

                    # Emission Log Probability
                    emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

                    # Transition Log Probability using bigrams
                    prev_tag_idx = self.tag2idx[new_tag_sequence[-2]]
                    transition_log_prob = np.log(self.bigrams[prev_tag_idx, tag_idx] + 1e-10)

                    new_log_prob = curr_log_prob + emission_log_prob + transition_log_prob

                    all_next_candidates.append({'tag': new_tag_sequence, 'log_prob': new_log_prob})

            # Retain only the top k of these candidates
            all_next_candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            candidates = all_next_candidates[:k]

        best_tags = candidates[0]['tag']
        # Return the best tags
        return best_tags

    def __beam_search_trigrams(self, sequence):
        k = 20  # Number of candidates to keep at each step

        # Pad the sequence with '^' at the beginning
        sequence = ['^'] + sequence

        # Initialize k candidate sequences
        candidates = [{'tag': ['^', 'O'], 'log_prob': 0.0}]

        for i in range(2, len(sequence)):
            all_next_candidates = []

            for curr_candidate in candidates:
                curr_tag = curr_candidate['tag']
                curr_log_prob = curr_candidate['log_prob']

                for possible_tag in self.tag2idx.keys():
                    if possible_tag == '^':
                        continue

                    new_tag_sequence = curr_tag + [possible_tag]

                    word_idx = self.__handle_unknown_word(sequence[i], None, None)
                    tag_idx = self.tag2idx[possible_tag]

                    # Emission Log Probability
                    emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

                    # Transition Log Probability
                    prev_tag_idx1 = self.tag2idx[new_tag_sequence[-3]]
                    prev_tag_idx2 = self.tag2idx[new_tag_sequence[-2]]
                    transition_log_prob = np.log(self.trigrams[prev_tag_idx1, prev_tag_idx2, tag_idx] + 1e-10)

                    new_log_prob = curr_log_prob + emission_log_prob + transition_log_prob

                    all_next_candidates.append({'tag': new_tag_sequence, 'log_prob': new_log_prob})

            # Retain only the top k of these candidates
            all_next_candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            candidates = all_next_candidates[:k]

        best_tags = candidates[0]['tag']
        # Return the best tags starting from 'O'
        return best_tags[1:]

    def __beam_search_fourgrams(self, sequence):
        k = 20  # Number of candidates to keep at each step

        # Pad the sequence with '^' at the beginning
        sequence = ['^', '^'] + sequence

        # Initialize k candidate sequences
        candidates = [{'tag': ['^', '^', 'O'], 'log_prob': 0.0}]

        for i in range(3, len(sequence)):
            all_next_candidates = []

            for curr_candidate in candidates:
                curr_tag = curr_candidate['tag']
                curr_log_prob = curr_candidate['log_prob']

                for possible_tag in self.tag2idx.keys():
                    if possible_tag == '^':
                        continue

                    new_tag_sequence = curr_tag + [possible_tag]

                    word_idx = self.__handle_unknown_word(sequence[i], None, None)
                    tag_idx = self.tag2idx[possible_tag]

                    # Emission Log Probability
                    emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

                    # Transition Log Probability using fourgrams
                    prev_tag_idx1 = self.tag2idx[new_tag_sequence[-4]]
                    prev_tag_idx2 = self.tag2idx[new_tag_sequence[-3]]
                    prev_tag_idx3 = self.tag2idx[new_tag_sequence[-2]]
                    transition_log_prob = np.log(
                        self.fourgrams[prev_tag_idx1, prev_tag_idx2, prev_tag_idx3, tag_idx] + 1e-10)

                    new_log_prob = curr_log_prob + emission_log_prob + transition_log_prob

                    all_next_candidates.append({'tag': new_tag_sequence, 'log_prob': new_log_prob})

            # Retain only the top k of these candidates
            all_next_candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            candidates = all_next_candidates[:k]

        best_tags = candidates[0]['tag']
        # Return the best tags starting from 'O'
        return best_tags[2:]

    def __greedy_search_unigrams(self, sequence):
        result_tags = ['O']
        cumulative_log_prob = 0.0  # Initialize the cumulative log probability

        for i in range(1, len(sequence)):
            max_prob = -np.inf
            best_tag = None

            for possible_tag in self.tag2idx.keys():  # Loop over all possible tags
                if possible_tag == '^':
                    continue

                word_idx = self.__handle_unknown_word(sequence[i], None, None)
                tag_idx = self.tag2idx[possible_tag]

                # Emission Log Probability
                emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

                if emission_log_prob > max_prob:
                    max_prob = emission_log_prob
                    best_tag = possible_tag

            result_tags.append(best_tag)

        # Append the <STOP> token to the result
        return result_tags

    def __greedy_search_bigrams(self, sequence):
        result_tags = ['O']
        cumulative_log_prob = 0.0  # Initialize the cumulative log probability

        for i in range(1, len(sequence)):
            max_prob = -np.inf
            best_tag = None

            for possible_tag in self.tag2idx.keys():  # Loop over all possible tags
                if possible_tag == '^':
                    continue

                word_idx = self.__handle_unknown_word(sequence[i], None, None)
                tag_idx = self.tag2idx[possible_tag]

                # Emission Log Probability
                emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

                # Transition Log Probability
                prev_tag_idx = self.tag2idx[result_tags[-1]]
                transition_log_prob = np.log(self.bigrams[prev_tag_idx, tag_idx] + 1e-10)

                # Calculate the sum of the cumulative probability and the incremental probability for the new tag
                current_sequence_prob = cumulative_log_prob + emission_log_prob + transition_log_prob

                if current_sequence_prob > max_prob:
                    max_prob = current_sequence_prob
                    best_tag = possible_tag

            result_tags.append(best_tag)
            cumulative_log_prob = max_prob  # Update the cumulative log probability

        # Append the <STOP> token to the result
        result_tags[-1] = '<STOP>'
        return result_tags

    def __greedy_search_trigrams(self, sequence):
        result_tags = ['^', 'O']

        # Pad the sequence with '^' at the beginning
        sequence = ['^'] + sequence

        cumulative_log_prob = 0.0  # Initialize the cumulative log probability

        for i in range(2, len(sequence)):
            max_prob = -np.inf
            best_tag = None

            for possible_tag in self.tag2idx.keys():  # Loop over all possible tags
                if possible_tag == '^':
                    continue

                word_idx = self.__handle_unknown_word(sequence[i], None, None)
                tag_idx = self.tag2idx[possible_tag]

                # Emission Log Probability
                emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

                # Transition Log Probability
                prev_tag_idx1 = self.tag2idx[result_tags[-2]]
                prev_tag_idx2 = self.tag2idx[result_tags[-1]]
                transition_log_prob = np.log(self.trigrams[prev_tag_idx1, prev_tag_idx2, tag_idx] + 1e-10)

                # Calculate the sum of the cumulative probability and the incremental probability for the new tag
                current_sequence_prob = cumulative_log_prob + emission_log_prob + transition_log_prob

                if current_sequence_prob > max_prob:
                    max_prob = current_sequence_prob
                    best_tag = possible_tag

            result_tags.append(best_tag)
            cumulative_log_prob = max_prob  # Update the cumulative log probability

        # Append the <STOP> token to the result
        result_tags[-1] = '<STOP>'
        return result_tags[1:]  # Return the tags starting from 'O'

    def __greedy_search_fourgrams(self, sequence):
        result_tags = ['^', '^', 'O']

        sequence = ['^', '^'] + sequence

        cumulative_log_prob = 0.0  # Initialize the cumulative log probability

        for i in range(3, len(sequence)):
            max_prob = -np.inf
            best_tag = None

            for possible_tag in self.tag2idx.keys():  # Loop over all possible tags
                if possible_tag == '^':
                    continue

                word_idx = self.__handle_unknown_word(sequence[i], None, None)
                tag_idx = self.tag2idx[possible_tag]

                # Emission Log Probability
                emission_log_prob = np.log(self.emissions[tag_idx, word_idx] + 1e-10)

                # Transition Log Probability
                prev_tag_idx1 = self.tag2idx[result_tags[-3]]
                prev_tag_idx2 = self.tag2idx[result_tags[-2]]
                prev_tag_idx3 = self.tag2idx[result_tags[-1]]
                transition_log_prob = np.log(
                    self.fourgrams[prev_tag_idx1, prev_tag_idx2, prev_tag_idx3, tag_idx] + 1e-10)

                # Calculate the sum of the cumulative probability and the incremental probability for the new tag
                current_sequence_prob = cumulative_log_prob + emission_log_prob + transition_log_prob

                if current_sequence_prob > max_prob:
                    max_prob = current_sequence_prob
                    best_tag = possible_tag

            result_tags.append(best_tag)
            cumulative_log_prob = max_prob  # Update the cumulative log probability

        # Append the <STOP> token to the result
        result_tags[-1] = '<STOP>'
        return result_tags[2:]  # Return the tags starting from 'O'

    def gold_sequence_prob(self, dev_data, gram):
        prob = []
        sentences, tags = dev_data
        for i in range(len(sentences)):
            prob.append(self.sequence_probability(sentences[i], tags[i], gram))
        return np.mean(prob)


def evaluate_all():
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")

    grams = [1,2,3,4]
    smooth_methods = ['k', 'l', 'no']
    algorithms = ['greedy search', 'beam search', 'viterbi']

    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ The following evaluation are using Add-K-Smoothing @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    pos_tagger_k = POSTagger()
    pos_tagger_k.train(train_data, smooth_methods[0])
    pos_tagger_k.train_or_load_UKN_classifier()
    for gram in grams:
        print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< evaluating {gram}-grams model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for algorithm in algorithms:
            print(f"%%%%%%%%%%% {gram}-grams with {algorithm} %%%%%%%%%%%")
            if gram == 4 and algorithm == 'viterbi':
                pass
            else:
                evaluate(dev_data, pos_tagger_k, gram, algorithm)

    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ The following evaluation are using Linear-Interpolation-Smoothing for trigram @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    pos_tagger_l = POSTagger()
    pos_tagger_l.train(train_data, smooth_methods[1])
    pos_tagger_l.train_or_load_UKN_classifier()
    gram = 3
    print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< evaluating {gram}-grams model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for algorithm in algorithms:
        print(f"%%%%%%%%%%% {gram}-grams with {algorithm} %%%%%%%%%%%")
        evaluate(dev_data, pos_tagger_l, gram, algorithm)

    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ The following evaluation are using No-Smoothing for trigram @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    pos_tagger_n = POSTagger()
    pos_tagger_n.train(train_data, smooth_methods[2])
    pos_tagger_n.train_or_load_UKN_classifier()
    gram = 3
    print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< evaluating {gram}-grams model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for algorithm in algorithms:
        print(f"%%%%%%%%%%% {gram}-grams with {algorithm} %%%%%%%%%%%")
        evaluate(dev_data, pos_tagger_n, gram, algorithm)


def save_outputs_to_file(filename, function_to_run, *args, **kwargs):
    # Backup the original standard output
    original_stdout = sys.stdout

    # Open the file in write mode and redirect stdout to the file
    with open(filename, 'w') as file:
        sys.stdout = file
        function_to_run(*args, **kwargs)

    # Reset stdout back to the original
    sys.stdout = original_stdout

if __name__ == "__main__":

    #save_outputs_to_file('evaluation/all_evaluation_result2.txt', evaluate_all)
    #evaluate_all()
    parser = argparse.ArgumentParser(description="POS Tagger Evaluation and Testing")
    parser.add_argument("-g", "--gram", type=int, choices=[1, 2, 3, 4],
                        help="Gram value (1, 2, 3, or 4), 4 doesn't support viterbi")
    parser.add_argument("-a", "--algorithm", choices=['greedy', 'beam', 'viterbi'],
                        help="Algorithm (greedy, beam, or viterbi)")
    parser.add_argument("-s", "--smooth", type=str, choices=['k', 'linear'], default='k',
                        help="Smoothing technique (either 'k' or 'linear')")
    parser.add_argument("-t", "--test_output_path", type=str,
                        help="Path to save test results, if desired")

    args = parser.parse_args()

    if args.gram == 4 and args.algorithm == "viterbi":
        raise ValueError("Using 4-gram with viterbi is not supported!")

    if args.gram != 3 and args.smooth != 'k':
        raise ValueError("Smoothing is only supported when gram=3!")

    print("Loading Data......")
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    print("Training......")
    pos_tagger = POSTagger()
    pos_tagger.train(train_data, args.smooth)
    pos_tagger.train_or_load_UKN_classifier()
    print("Done Training")

    print("Evaluating on Dev......")

    if args.algorithm == "greedy":
        alg = "greedy search"
    elif args.algorithm == "beam":
        alg = "beam search"
    else:
        alg = args.algorithm

    evaluate(dev_data, pos_tagger, args.gram, alg)


    if args.test_output_path:
        output_test(pos_tagger, args.gram, alg, args.test_output_path)





