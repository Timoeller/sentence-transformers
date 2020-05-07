from . import InputExample
import os
import numpy as np
import random
from tqdm import tqdm


class myICTReader():
    """
    Reads in any file that is in standard Bert pretraining format
    Needs separate sentences per line
    Documents are separated by a single blank line
    """
    def __init__(self, dataset_folder, prob_negative_example = 0.5):
        self.dataset_folder = dataset_folder
        self.prob_negative_example = prob_negative_example

    def get_examples(self, filename, max_examples=0):
        """
        Gets the examples from the given filename

        :param filename: str, file that should be converted to ICT training data
                             - The file should be a text file in Bert pretraining format: one sentence per line, documents separated by a single newline
                             - Example format is in this archive: https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/lm_finetune_nips.tar.gz
        """
        data = open(os.path.join(self.dataset_folder, filename),mode="rt", encoding="utf-8").readlines()

        doc_boundaries = np.array([1 if x == "\n" else 0 for x in data])
        doc_ids = np.cumsum(doc_boundaries)
        examples = []
        id = 0
        for i,line in tqdm(enumerate(data), desc="Loading Training data", total=len(data)):
            #skipping first and last sentence in data
            if i == 0 or i >= len(data)-1:
                continue
            # skipping first and last sentence in document, because we cannot construct
            # the context with prev and following sentence
            if np.sum(doc_boundaries[[i-1,i+1]]) > 0:
                continue

            sentence = line
            if random.random() > self.prob_negative_example:
                context = data[i-1] + " " + data[i+1]  # TODO: potential improvement - when sentence and context have large word overlap, dismiss this instance
                label = "context"
            else:
                context = self._get_random_context(data, doc_boundaries, doc_ids, forbidden_doc=doc_ids[i])
                label = "random"

            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence, context], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"context": 0, "random": 1}

    def _get_random_context(self, data, doc_boundaries, doc_ids, forbidden_doc):
        found = False
        while not found:
            idx_cand = np.random.randint(1,len(data)-1)
            if np.sum(doc_boundaries[[idx_cand-1,idx_cand+1]]) > 0:
                continue
            if doc_ids[idx_cand] == forbidden_doc:
                continue
            context = data[idx_cand-1] + " " + data[idx_cand+1]
            found = True
        return context

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]