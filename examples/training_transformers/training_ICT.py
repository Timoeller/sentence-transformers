from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers.ICTReader import myICTReader
from sentence_transformers.readers.STSDataReader import STSBenchmarkDataReader
import logging
from datetime import datetime
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

# Read the dataset
batch_size = 32
# the lm_finetune_nips nips dataset needs to be downloaded from https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/lm_finetune_nips.tar.gz
ict_reader = myICTReader('../datasets/lm_finetune_nips')
sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark') # this dataset is available through examples/datasets/get_data.py
train_num_labels = ict_reader.get_num_labels()
# the resulting model, e.g. "training_ict_bert-base-uncased-2020-05-07_15-48-42/0_Transformer" can be loaded into FARM or Huggingface as Language Model
model_save_path = 'output/training_ict_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Convert the dataset to a DataLoader ready for training
train_data = SentencesDataset(ict_reader.get_examples('dev.txt',max_examples=6000), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

# load dev evaluator with sentence similarity task sts-dev
# Finetuning Sentence Transformers with inverse cloze task + nips papers (does not seem to improve sentence similarity by much):
# Without finetuning (vanilla bert) : Pearson: 0.5917	Spearman: 0.5932
# With ICT finetuning after 3k steps: Pearson: 0.6574	Spearman: 0.6809
# longer training does not improve the sentence similiarity
logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
num_epochs = 1

warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

# load model and test again on dev
model = SentenceTransformer(model_save_path)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
model.evaluate(evaluator)
