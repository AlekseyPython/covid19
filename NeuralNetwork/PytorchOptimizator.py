import math
import time
import nltk
import functools
import itertools
import multiprocessing
from collections import Counter, OrderedDict
import pandas as pd
import torch
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
from Business.AMachineLearning import AOptimizatorNeuralNetwork
from .Embeddings import Embeddings
from .LearningData import PreparedData, LearningData
from .PyTorchNetwork import PyTorchNetwork

     
class PytorchOptimizator(AOptimizatorNeuralNetwork):
    def __init__(self, source_data, converted_data):   
        AOptimizatorNeuralNetwork.__init__(self)
        
        # Hyperparameters
        self.EPOCHS     = 10 
        self.LR         = 1e-3
        self.L2         = 5e-4
        self.BATCH_SIZE = 1024
        self.DROP_OUT   = 0.3
        self.NUM_WORKERS= multiprocessing.cpu_count()
        
        #this coefficient shows, that the error to classify a weaker course of the disease as a stronger one
        #is more important for us, than the reverse
        self.multiplier_of_zero_class   = 2
        
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.BCELoss()
        
        self.prepared_data = PreparedData(source_data, converted_data)
    
    @staticmethod
    def _get_words(element_data_set):
        return nltk.word_tokenize(element_data_set[0])
    
    @staticmethod
    def _create_vocab(train_dataset):
        with multiprocessing.Pool() as pool:
            structured_words = pool.map(PytorchOptimizator._get_words, train_dataset)
        
        plain_words = list(itertools.chain.from_iterable(structured_words))
        counter = Counter(plain_words)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        vocabulary = vocab(ordered_dict, min_freq=30)
        
        unk_token = '<unk>'
        default_index = -1
        if unk_token not in vocabulary: 
            vocabulary.insert_token(unk_token, 0)
        vocabulary.set_default_index(default_index)
        return vocabulary
    
    def _create_model(self, vocabulary):
        vocab_size = len(vocabulary)
        embed_dim  = 32
        hidden_dim = 16 
        dropout    = self.DROP_OUT 
        model      = PyTorchNetwork(vocab_size, embed_dim, hidden_dim, dropout).to(self.device)
        return model
    
    def _print_model(self, model, vocabulary):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(model)
        print(f'The model has {count_parameters(model):,} trainable parameters')
         
    def learn(self, severity_of_disease, silently):
        embeddings = Embeddings()
        model_embeddings = embeddings.get_model()
        return
        
        train_dataset = LearningData(self.prepared_data, severity_of_disease, train=True)
        test_dataset  = LearningData(self.prepared_data, severity_of_disease, train=False)
       
        vocabulary = self._create_vocab(train_dataset)
        model      = self._create_model(vocabulary)
        
        if not silently:
            self._print_model(model, vocabulary)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR, weight_decay=self.L2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9, verbose=not silently)
        
        collate_batch = functools.partial(self._collate_batch, vocabulary)
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS,
                                      shuffle=True, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS,
                                     shuffle=True, collate_fn=collate_batch)
        
        previous_Kappa, previous_MCC = None, None
        for epoch in range(1, self.EPOCHS + 1):
            epoch_start_time = time.time()
            
            self._train(train_dataloader, model, optimizer)
            Kappa, MCC = self._evaluate(test_dataloader, model)
            
            if previous_Kappa is not None and previous_Kappa>Kappa or previous_MCC>MCC:
                scheduler.step()
            else:
                previous_Kappa = Kappa
                previous_MCC   = MCC
                
            if not silently:
                print('-' * 59)
                print('| end of epoch {:3d} | time: {:5.2f}s | Kappa: {:3.3f}   MCC: {:3.3f}'.
                      format(epoch, time.time() - epoch_start_time, Kappa, MCC))
                print('-' * 59)
            
    def _train(self, dataloader, model, optimizer):
        model.train()
        for texts, targets, offsets in dataloader:
            texts, targets, offsets = texts.to(self.device), targets.to(self.device), offsets.to(self.device)
            
            optimizer.zero_grad()
            predictions = model(texts, offsets)
            loss        = self.criterion(predictions, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
    
    def _evaluate(self, dataloader, model):
        model.eval()
        
        all_Kappa, all_MCC = 0, 0
        needed_metrics = set(['Kappa', 'MCC'])
        with torch.no_grad():
            for texts, targets, offsets in dataloader:
                predictions = model(texts, offsets)
                
                metrics = self._count_metrics(predictions, targets, needed_metrics)
                all_Kappa += metrics['Kappa']
                all_MCC   += metrics['MCC']
        return all_Kappa, all_MCC 
    
    def _collate_batch(self, vocabulary, batch):
        text_pipeline = lambda x: vocabulary(nltk.word_tokenize(x))

        label_list, text_list, offsets = [], [], [0]
        for current_text, current_label in batch:
            label_list.append(current_label)
            processed_text = torch.tensor(text_pipeline(current_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
            
        label_list = torch.tensor(label_list, dtype=torch.float32)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        
        # return text_list.to(self.device), label_list.to(self.device), offsets.to(self.device)
        return text_list, label_list, offsets
    
    def _count_metrics(self, predictions, targets, needed_metrics):
        def get_consistent_number_intervals():  
            optimun_quantity_intervals = None
            for quantity_intervals in range(2, 100):
                size_interval = 1 / quantity_intervals
                df['interval'] = df['predictions'] // size_interval 
            
                grouped = df['targets'].groupby(df['interval'])
                means = grouped.mean()
                
                exist_violation = False
                previous_target = 0
                for interval in means.index:
                    target = means.loc[interval]
                    if target <= previous_target:
                        exist_violation = True
                        break
                    previous_target = target
                
                if exist_violation:
                    optimun_quantity_intervals = quantity_intervals-1
                    break
                
            if optimun_quantity_intervals is None:
                raise RuntimeError('It is impossible to find the optimal number of intervals for these predictions!')
            return optimun_quantity_intervals
    
        predictions = predictions.cpu()
        targets     = targets.cpu()
        df = pd.DataFrame({'predictions':predictions, 'targets':targets})
        
        optimun_quantity_intervals = get_consistent_number_intervals()
        size_interval = 1 / optimun_quantity_intervals
        df['interval'] = df['predictions'] // size_interval
        
        grouped = df['targets'].groupby(df['interval'])
        means = grouped.mean()
        
        middle_probability = self.multiplier_of_zero_class / (self.multiplier_of_zero_class + 1)
        left_series = means[means<=middle_probability]
        right_series = means[means>middle_probability]
        
        if len(left_series)==0 or len(right_series)==0:
            middle_point = middle_probability
        else:
            min_interval = left_series.index[-1]
            min_point = min_interval * size_interval 
            min_probability = means[min_interval]
            
            max_interval = right_series.index[0]
            max_point = max_interval * size_interval
            max_probability = means[max_interval]
            
            derivative = (max_probability-min_probability) / (max_point-min_point)
            
            if middle_probability - min_probability < max_probability - middle_probability:
                middle_point = min_point + (middle_probability-min_probability) / derivative
            else:
                middle_point = max_point - (max_probability-middle_probability) / derivative
                
        TP = df[(df.predictions>middle_point) & (df.targets==True)].shape[0]
        TN = df[(df.predictions<=middle_point) & (df.targets==False)].shape[0]
        FP = df[(df.predictions>middle_point) & (df.targets==False)].shape[0]
        FN = df[(df.predictions<=middle_point) & (df.targets==True)].shape[0]
        All = TP+TN+FP+FN
        
        counted_metrics = {}
        if 'Accuracy' in needed_metrics:
            Accuracy = (TP + TN) / All
            counted_metrics['Accuracy'] = round(100 * Accuracy, 3)
        
        if 'Balanced accuracy' in needed_metrics:
            if (TP+FN)==0 or (TN+FP)==0:
                Balanced_Accuracy = 0
            else:
                Balanced_Accuracy = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
                Balanced_Accuracy = round(100 * Balanced_Accuracy, 3)
            counted_metrics['Balanced accuracy'] = Balanced_Accuracy    
        
        if 'Kappa' in needed_metrics:
            Accuracy = (TP+TN) / All
            Accuracy_chance = (TN+FP)*(TN+FN)/(All*All) + (TP+FN)*(TP+FP)/(All*All)
            Kappa = (Accuracy-Accuracy_chance) / (1 - Accuracy_chance)
            counted_metrics['Kappa'] = round(100 * Kappa, 3)    
                
        if 'MCC' in needed_metrics:
            if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) == 0:
                MCC = 0
            else:
                MCC = (TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                MCC = round(100 * MCC, 3)
            counted_metrics['MCC'] = MCC    
                
        return counted_metrics
    
    