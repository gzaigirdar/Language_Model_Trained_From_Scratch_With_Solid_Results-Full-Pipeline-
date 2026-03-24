### Project Description
This project focuses on training a decoder-only transformer language model from scratch under limited computational resources and dataset size. The objective is to generate coherent text and handle basic conversational tasks.

The repository provides a complete end-to-end pipeline, including:
Data collection and preprocessing
Tokenization
Model architecture implementation (low-level PyTorch)
Training setup and utilities
Evaluation and inference
Experiments

Three separate experiments were conducted, each training an independent model with its own dataset, hyperparameters, and configuration:
Casual Chat – informal dialogue generation
Tiny Chat – short conversational response generation
Story Generator – generates short stories from a prompt

### Dataset and Tokenizer
Three separate datasets were constructed for the experiments, each corresponding to one model (Casual Chat, Tiny Chat, Story Generator). All datasets were sourced from publicly available corpora and processed using a consistent pipeline.

Sources
Hugging Face datasets (e.g., TinyChat, TinyStories, conversational datasets)
Kaggle dataset(s)
Small custom-added samples
Preprocessing

Each dataset was standardized using the same steps:

Replace role markers with unified tokens (e.g., <user>, <bot>)
Convert text to lowercase
Trim unnecessary whitespace
Separate punctuation (. ? ! , ;) into individual tokens
Add sequence boundary tokens: <start> and <end>
Tokenization

All datasets were tokenized using the T5-small SentencePiece tokenizer.

#### Computinal Resources:
Gpu: RTX 4060 8 GB, System RAM: 32 GB , Processor: Intel Core i5-13400F (16 logical cores, up to 4.6 GHz) OS: Linux Ubuntu

#### Evaluation metrics and methods:

Both training and validation splits to monitor learning behavior and overfitting.
Perplexity to measure token prediction quality.
Custom prompts to evaluate coherence and generation quality

#### Training techniques applied:

Dropout
Weight decay
Learning rate scheduling
bfloat16 precision (for faster training)
Gradient clipping (for stability)

#### Detailed experiment configurations and results are available in:
Full details about the expeiments regarding training,dataset preprocessing, evaluation,results,inference can be found in Report.pdf
All Training Notebooks can be found /Training Notebooks
All Evaluation Notebooks can be found in /Evaluation Notebooks 

#### How To Use The Trained Models:

Create and activate  virtual environment:
python -m venv .venv 
source .venv/bin/activate
Run inference script for one of the models:
    Casual Chat:
        python Casual_chat_Inf.py
    Tiny Chat:
        python Tiny_Chat_Inf.py 
    Story Gen:
        python Story_Gen_Inf.py
Additionally, Top_K and Temperature can be passed in the command line for each script.
Example: python Casual_chat_Inf.py –Top_K 20 –Temp 0.60
(double dash before the parameter name)


#### Results:
All three experiments produced strong results. Each model was able to generate coherent text and perform according to its specific training objective. Casual Chat and Tiny Chat successfully handled conversational prompts, while Story Generator produced short, contextually relevant stories. Minor inconsistencies occasionally appeared due to limited dataset size and training resources, but overall, each model demonstrated the ability to generate task-aligned, meaningful output. 


##### Casual Chat output on the Custom Test prompt:

Sample Outputs from the MainModel: here are some response to basic dialogue with greedy decoding

| Prompt                               | Response                                                |
| ------------------------------------ | ------------------------------------------------------- |
| how are you ?                        | i'm fine. how about you?                                |
| what is your name ?                  | my name is li lin.                                      |
| what time is it now ?                | it's 7 : 00 p. m.                                       |
| hey , what are you up to ?           | i'm just a little tired.                                |
| hey , what's up ?                    | nothing much.                                           |
| where do you work ?                  | i work at a bank.                                       |
| do you like to watch sports ?        | i like watching football.                               |
| where are you from ?                 | i'm from taiwan.                                        |
| do you like chinese food ?           | yes, i do. i love it.                                   |
| are you a chatbot ?                  | yes, i am.                                              |
| where do you live ?                  | i live in san francisco.                                |
| do you want to go get some food ?    | i'd like to, but i'm not sure if i'll get enough sleep. |
| do you want to watch a movie ?       | i'd love to.                                            |
| tell me about yourself .             | i'm a teacher.                                          |
| i'm doing well . thanks for asking . | you're welcome.                                         |

The model demonstrates an understanding of sentence structure and grammar, producing mostly meaningful responses, but its capabilities are constrained by the dataset and model size.

-----------------------------------------------------------------------------------------------------

##### Tiny Chat outputon the Custom Test prompt::

    | Input                                                | Output                                                                              |
    | ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
    | hey how are you?                                     | i am feeling very happy today, thank you for asking.                                |
    | what is the meaning of love?                         | love is a strong feeling that brings people together in many ways.                  |
    | do you like watching movies?                         | i enjoy watching movies that have a wide range of interesting characters and plots. |
    | do you like reading books or watching movies better? | i enjoy reading books because they take me to different worlds.                     |
    | i'm doing well, how about you?                       | i am feeling a bit bad today, thank you for asking.                                 |

The model demonstrated strong fluency in grammar and generated meaningful responses; however, it struggled with very informal dialogue, likely because the TinyChat dataset mostly contains formal conversations.
__________________________________________________________________________________________________________________________________________________

##### Story Generator Model's response on the Custom Test prompt:

Prompt 
once upon a time there was this planet

Story:
Once upon a time there was this planet. It was very big and very big. One day, a little girl named Lucy wanted to explore the planet. She asked her mom if she could go. Her mom said yes, so Lucy ran outside. She saw the big planet and it was very big. She wanted to explore it. So Lucy started to climb the big planet. She climbed and climbed until she reached the top. She looked around and saw lots of things. She saw a big tree and some birds flying in the sky. She was so happy! Lucy kept climbing until she reached the top. She looked around and saw the big planet. She was so excited! She wanted to explore it. So she climbed down the big planet and ran back home. She had a lot of fun exploring the big planet.

Prompt: there was a storm

Story:  
There was a storm. It was very loud and scary. The wind was blowing and the rain was pouring down. The wind was so strong that it made the rain drop down. The rain was so strong that it made the ground shake. The rain was so strong that it made the ground shake even harder. The rain kept falling and the ground shook even harder. The rain kept falling and the ground shook even harder. The rain kept falling and the ground shook even harder. The rain kept falling and the ground shook even harder. The rain kept falling and the ground shook until it was so heavy that it was almost like it was shaking. The rain stopped and the sun came out. The ground was still shaking. The storm was so strong that it made the ground shake even harder. The rain stopped and the ground was quiet again. The sun came out and the ground was dry again.


The model generates fully coherent and grammatically correct sentences with a reasonable narrative, but it is limited to short and simple stories, as the Tiny Stories dataset consists of brief children’s stories.

________________________________________________________________________________________________________________________________________________