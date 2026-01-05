How to use the model:
Create virtual  enviorment,Install all the packages packages,and  Run the infernce script to chat with the model.
Steps:

 - python -m venv .llmenv
 - source .llmvenv/bin/activate
 - pip install requirement.txt 
 - python infernce.py 

Project Description

This project aims to train a transformer-based decoder-only model from scratch to generate coherent text and respond to basic, general dialogues.
The repository provides a complete, modular pipeline for training a language model, including data preprocessing, tokenization, model architecture implementation, training tools and environment setup, training, evaluation, and inference. Its organized structure makes it easy to train language models with minimal setup.

Dataset and Tokenizer

The training data consists of multiple small datasets, including ConvAI, General_Conversation_Mixed_Dataset, everyday-conversations-llama3.1-2k, and 100k samples from TinyChat, all sourced from Hugging Face. Additionally, one dataset was obtained from Kaggle, and small custom hardcoded samples were added.(Note: check data.info for more details on dataset and sources!)
Full dataset details can be found in dataset.info within the data building and loading folder.
(Note: For clarity, this collection is referred to as the combined dataset.)
Google's T5 small tokenizer from Hugging Face was chosen for its effective handling of empty and trailing spaces.

Motivation

The goal of this project was to build a language model entirely from scratch capable of generating coherent text. Most existing resources do not show clear way to train a model from scartch to generate reasonable ouput. While I have prior experience with transformer architectures and training, this is my first complete training a transformer based  model from scartch, which helped solidify my understanding of language models and transformer mechanics.

Experiment Details

The model was trained for 5 epochs on 200k samples using:

    Cosine learning rate scheduler with a base learning rate of 0.001

    bfloat16 precision for faster training

    Gradient clipping to stabilize training

Full experiment details are available in experiment.doc within the repository.
(Note: The training workflow is documented in Best_Model_80t.ipynb.)

Results

A 41M-parameter model was trained on 200k structured dialogue samples for 5 epochs. Despite the limited dataset size, the model achieved a final perplexity of 6 and demonstrated the ability to generate coherent and meaningful responses to prompts. Sample outputs can be found at the end of the notebook.
(Note: Evaluation details are in evaluation.ipynb.)

The model architecture was implemented from scratch using low-level PyTorch operations, following the GPT-2 design with minor modifications, such as using RMSNorm instead of LayerNorm. (Folder: DecoderOnly_Transformer_Custom_implementation)


Additionally topk and temp values can set to for diverse output by passing them in the command line
python infernce.py --Top_K=3 --Temp=0.30


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




Other Experiments: There were other experiments were run such as trainging a 41m model to generate short stories using tiny stories dataset, which showed strong result as well.The tinychat stories model were fined tuned with combined dataset to generate convo. The third experiment was trainig model using 1 millon samples of tiny chat(without any other sourc), which showed very strong flunecy and meaning full response to fomarl dialogue, but it didn't to well as well combined dataset model in general everyday dialouge.


Tiny chat model can be run using:

python Tiny_chat_inf.py 

Tiny story generator can be run using:

python gen_story_inf.py


________________________________________________________________________________________________________________________________________________

Tiny Chat Model output:

    | Input                                                | Output                                                                              |
    | ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
    | hey how are you?                                     | i am feeling very happy today, thank you for asking.                                |
    | what is the meaning of love?                         | love is a strong feeling that brings people together in many ways.                  |
    | do you like watching movies?                         | i enjoy watching movies that have a wide range of interesting characters and plots. |
    | do you like reading books or watching movies better? | i enjoy reading books because they take me to different worlds.                     |
    | i'm doing well, how about you?                       | i am feeling a bit bad today, thank you for asking.                                 |

The model demonstrated strong fluency in grammar and generated meaningful responses; however, it struggled with very informal dialogue, likely because the TinyChat dataset mostly contains formal conversations
__________________________________________________________________________________________________________________________________________________

Tiny Stories Model ouput:

Prompt 
once upon a time there was this planet

Story:
Once upon a time there was this planet. It was very big and very big. One day, a little girl named Lucy wanted to explore the planet. She asked her mom if she could go. Her mom said yes, so Lucy ran outside. She saw the big planet and it was very big. She wanted to explore it. So Lucy started to climb the big planet. She climbed and climbed until she reached the top. She looked around and saw lots of things. She saw a big tree and some birds flying in the sky. She was so happy! Lucy kept climbing until she reached the top. She looked around and saw the big planet. She was so excited! She wanted to explore it. So she climbed down the big planet and ran back home. She had a lot of fun exploring the big planet.

Prompt: there was a storm

Story:  
There was a storm. It was very loud and scary. The wind was blowing and the rain was pouring down. The wind was so strong that it made the rain drop down. The rain was so strong that it made the ground shake. The rain was so strong that it made the ground shake even harder. The rain kept falling and the ground shook even harder. The rain kept falling and the ground shook even harder. The rain kept falling and the ground shook even harder. The rain kept falling and the ground shook even harder. The rain kept falling and the ground shook until it was so heavy that it was almost like it was shaking. The rain stopped and the sun came out. The ground was still shaking. The storm was so strong that it made the ground shake even harder. The rain stopped and the ground was quiet again. The sun came out and the ground was dry again

Prompt: 
In a magical jungle, animals could talk to each other

Story:
In a magical jungle, animals could talk to each other. One day, a little monkey named Jack was walking through the jungle. He heard a strange noise coming from the trees. He stopped and listened carefully. He heard a voice coming from the trees. It said, "Hello, Jack! I'm a monkey. I'm looking for something to eat." Jack was surprised. He asked, "What are you looking for?" The monkey replied, "I'm looking for some food. I'm looking for some." Jack smiled and said, "I can help you. Follow me!" So Jack and the monkey went on an adventure. They walked through the jungle, looking for food. After a while, they found a big, juicy apple. Jack said, "This apple is so delicious! Thank you, monkey!" The monkey smiled and said, "You're welcome, Jack. I'm glad I could help." And with that, Jack and the monkey went back home.



The model generates fully coherent and grammatically correct sentences with a reasonable narrative, but it is limited to short and simple stories, as the Tiny Stories dataset consists of brief children’s stories.

________________________________________________________________________________________________________________________________________________