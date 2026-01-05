import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup


class Train_tools:
    def __init__(self,config):
        self.LR = config['learning_rate']
      
        self.LabelSmoothing = config['Label_smoothing']
        self.GradAccumulationSteps = config['GradAccumulation_steps']
        self.WeightDecay = config['Weight_decay']
        self.WarmupStepsPerc = config['warmupsteps_percentage']
        self.Epochs = config['Epochs']
        self.NumBatches = config['Num_batches']
        self.TotalTrainingSteps = (self.NumBatches // self.GradAccumulationSteps) * self.Epochs
        self.WarmUpSteps = int(self.TotalTrainingSteps * self.WarmupStepsPerc)
        self.ignore_index = config['ignore_index']
        self.Loss_fn = nn.CrossEntropyLoss(label_smoothing=self.LabelSmoothing,ignore_index=self.ignore_index)
        self.LRScheduler = None
        self.Optimizer = None

        


    
    def getTools(self,model_prams):
        self.Optimizer = AdamW(model_prams,lr=self.LR,weight_decay=self.WeightDecay)
        self.LRScheduler = get_cosine_schedule_with_warmup(
            self.Optimizer,
            num_warmup_steps=self.WarmUpSteps,
            num_training_steps=self.TotalTrainingSteps,
            num_cycles=0.50,
            last_epoch=-1

        )
        '''self.LRScheduler = get_linear_schedule_with_warmup(
            self.Optimizer,
            num_warmup_steps=self.WarmUpSteps,
            num_training_steps= self.TotalTrainingSteps
        )'''
        return self.Loss_fn,self.Optimizer,self.LRScheduler 
    

