import mxnet as mx
import gluonnlp as nlp

model_name = 'bert_12_768_12'
dataset_name = 'book_corpus_wiki_en_uncased'
_, vocab = gluonnlp.model.get_model(model_name,
                                    dataset_name=dataset_name,
                                    pretrained=False, root='Gluonnlp/model')