
Documentation at:
https://huggingface.co/docs
https://huggingface.co/docs/transformers/index

# Chapter 1: Transformers, what can they do?

## A. Transformers: 

The most basic object in the 🤗 Transformers library is the pipeline() function. It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer.

Transformers can be grouped into three categories:
* GPT-like (also called auto-regressive Transformer models) : GPT-2 was trained for causal language modeling
* BERT-like (also called auto-encoding Transformer models) : BERT was trained for mask filling
* BART/T5-like (also called sequence-to-sequence Transformer models)

Compared to previous instances of machine learning, all those transformers have been trained as **language models**. Meaning they have been trained on large amounts of raw texte in a self-supervised fashion.
Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model. That means that humans are not needed to label the data.

This type of model develops a statistical understanding of the language it has been trained on, but it’s not very useful for specific practical tasks. 
Because of this, the general pretrained model then goes through a process called transfer learning. 
During this process, the model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task.

An example of a task is predicting the next word in a sentence having read the n previous words. This is called **causal language modeling** because the output depends on the past and present inputs, but not the future ones.


## B. Pretraining

Pretraining is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.
This pretraining is usually done on very large amounts of data. Therefore, it requires a very large corpus of data, and training can take up to several weeks.


## C. Fine-tuning

Fine-tuning, on the other hand, is the training done after a model has been pretrained. To perform fine-tuning, you first acquire a pretrained language model, then perform additional training with a dataset specific to your task.

Why not simply train directly for the final task? There are a couple of reasons:
- The pretrained model was already trained on a dataset that has some similarities with the fine-tuning dataset. The fine-tuning process is thus able to take advantage of knowledge acquired by the initial model during pretraining (for instance, with NLP problems, the pretrained model will have some kind of statistical understanding of the language you are using for your task).
- Since the pretrained model was already trained on lots of data, the fine-tuning requires way less data to get decent results.
- For the same reason, the amount of time and resources needed to get good results are much lower.

For example, one could leverage a pretrained model trained on the English language and then fine-tune it on an arXiv corpus, resulting in a science/research-based model. The fine-tuning will only require a limited amount of data: the knowledge the pretrained model has acquired is “transferred,” hence the term transfer learning.

Fine-tuning a model therefore has lower time, data, financial, and environmental costs. It is also quicker and easier to iterate over different fine-tuning schemes, as the training is less constraining than a full pretraining.


## D. Transfer learning

Consists in leveraging the knowledge acquired by a model A on a specific task to a model B for a different task with a smaller dataset.
Training from scratch requires a lot of data and compute. Moreover, fine-tuning a pretrained model for a new task yields a higher accuracy in the new task, compared to the training task.

Usually, it is applied by dropping the head (last layers focused on the pretraining objective) and replacing it with a new randomly initialized head suitable for the task at hand.
For instance, on a mask filling model we drop the head and replace it with a classifier that has 2 outputs to perform a task that has 2 outputs.
To be as efficient as possible, the pretrained model used should be as similar as possible to the task it will be fine-tuned on (e.g. use a german pretrained model to classify german sentences).

However, mind that pretrained models transfer their knowledge as well as their biases.


## E. General architecture (encoders, decoders, encoder-decoders)

The Transformer architecture was originally designed for translation:
During training, the encoder receives inputs (sentences) in a certain language, while the decoder receives the same sentences in the desired target language. 
1. In the encoder, the attention layers can use all the words in a sentence (since, as we just saw, the translation of a given word can be dependent on what is after as well as before it in the sentence). 
2. The decoder, however, works sequentially and can only pay attention to the words in the sentence that it has already translated (so, only the words before the word currently being generated). 
For example, when we have predicted the first three words of the translated target, we give them to the decoder which then uses all the inputs of the encoder to try to predict the fourth word.

To speed things up during training (when the model has access to target sentences), the decoder is fed the whole target, but it is not allowed to use future words.

Transformers are based on the attention mechanism:
- The encoder accepts inputs that represent text. It converts this text (words) into numerical representations. These numerical representations are called **embeddings** or **features**. It uses a self-attention mechanism as main component. Its properties are bi-directional. This means that the model is optimized to acquire understanding from the input.
- The decoder is similar: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs. It accepts text inputs, use a masked-self-attention, is usually used in auto-regressive models and is uni-directional.
- Encoder-Decoder, or 'sequence-to-sequence transformer' is the combination of the two parts.


## E. Attention layers

A key feature of Transformer models is that they are built with special layers called attention layers.
This layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.

To put this into context, consider the task of translating text from English to French. Given the input “You like this course”, a translation model will need to also attend to the adjacent word “You” to get the proper translation for the word “like”, because in French the verb “like” is conjugated differently depending on the subject. The rest of the sentence, however, is not useful for the translation of that word. In the same vein, when translating “this” the model will also need to pay attention to the word “course”, because “this” translates differently depending on whether the associated noun is masculine or feminine. Again, the other words in the sentence will not matter for the translation of “this”. With more complex sentences (and more complex grammar rules), the model would need to pay special attention to words that might appear farther away in the sentence to properly translate each word.


## F. Carbon footprints

Transformers are big models with a huge carbon footprint, given :
* Type of energy (renewable vs non-renewable)
* Training time
* Hardware

-> Chose low computing instances
-> Chose pre-trained model when available
-> Fine-tuning the last layers instead of training from scratch
-> Starting with smaller experiments and debugging
-> Doing literature review to choose hyperparameters ranges
-> Random search vs Grid search for hyperparameters

This is why sharing language models is paramount: sharing the trained weights and building on top of already trained weights reduces the overall compute cost and carbon footprint of the community.  

By the way, you can evaluate the carbon footprint of your models’ training through several tools. For example ML CO2 Impact or Code Carbon which is integrated in 🤗 Transformers. To learn more about this, you can read this blog post which will show you how to generate an emissions.csv file with an estimate of the footprint of your training, as well as the documentation of 🤗 Transformers addressing this topic.


## G. Architecture-Checkpoints-Models:

* Architecture: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
* Checkpoints: These are the weights that will be loaded in a given architecture.
* Model: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both. This course will specify architecture or checkpoint when it matters to reduce ambiguity.

For example, BERT is an architecture while bert-base-cased, a set of weights trained by the Google team for the first release of BERT, is a checkpoint. However, one can say “the BERT model” and “the bert-base-cased model.”

* Auto-regressive models: the output variable depends linearly on its own previous values (and on a stochastic term); in other words, the next output is chosen based on the probabilities of each word, given all the words currently in the input.


## H. Encoder models

Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are often called auto-encoding models.

* Encoders input a text and output exactly one numerical representation for each word used as input. 
* This numerical representation is composed of sequences (aka vectors) of numbers per input word. This numerical representation is called a **feature vector** or **feature tensor**. The dimension of that vector is defined by the architecture of the model.
* It contain the value of a word, but contextualized. It takes into account the words around it, called **context**. This is done thanks to the **self-attention mechanism**. This mechanism relates to different positions or different words in a single sequence in order to compute a representation of that sequence. The representation of a word is affected by other words in a sequence

Encoders are:
* Bi-directional: context from the left and the right
* Good at extracting vectors that carry meaningful information about a sequence
* For tasks: Sequence classification (such as sentiment analysis), question answering, masked language modelling, named entity recognition
* Performing NLU: Natural Language Understanding
* Examples: BERT, RoBERTa, ALBERT

In other words, encoders are good at obtaining an understanding of sequences, and the relationship/interdependence between words. They are  best suited for tasks requiring an understanding of the full sentence.

The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.


## I. Decoder models

Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called auto-regressive models.

Just like encoders:
* Decoders input a text and output exactly one numerical representation for each word used as input.
* This numerical representation is composed of sequences (aka vectors) of numbers per input word. This numerical representation is called a **feature vector** or **feature tensor**. The dimension of that vector is defined by the architecture of the model.
* It contain the value of a word, but contextualized. It takes into account the words around it, called **context**. This is done thanks to the **self-attention mechanism**.

Where decoders differ from encoders:
* The **self-attention mechanism** is a **masked self attention**: words can only see the words on one single side, for instance the left, whereas the other side is hidden: for instance words to the right are absent from the context (i.e., the **right context** is masked).

Decoders are:
* Unidirectional: access to the left OR right context
* Can be used as stand-alone models and perform a wide variety of tasks, however with less strength given the task
* Great at Causal Language Modeling, causal tasks such as text generation (i.e., given a 1st known sequence, generates a new 2nd sequence then appends it to the known in order to generate the next 3rd sequence)
* Great at NLG: Natural Language Generation
* Examples: GPT-2, GPT Neo

These models can perform most of the same taks as encoder models, albeit with generally a loss in performances. They are best suited for tasks involving text generation.

The pretraining of decoder models usually revolves around predicting the next word in the sentence.


## J. Encoder-Decoder models

Encoder-decoder models (also called sequence-to-sequence models) use both parts of the Transformer architecture. 

At each stage:
* The attention layers of the encoder can access all the words in the initial sentence. 
* Whereas the attention layers of the decoder can only access the words positioned before a given word in the input.

To be more precise:
* At stage one:
    -> The encoder receives the initial sequence and outputs a representations
    -> The decoder receives the start of the initial sequence as well as the representations from the encoder and outputs a 1st word
* At stage two:
    -> The decoder receives the start of the initial sequence, the outputed 1st word as well as the representations from the encoder and outputs a 2nd word
* At stage three:
    -> The decoder receives the start of the initial sequence, the outputed 1st and 2nd word as well as the representations from the encoder and outputs a 3rd word
* Until a stopping value (e.g., a dot)

Where encoder-decoders differ from encoders and decoders:
* The encoder-decoder is separated into two components:
    * The encoder takes care of understanding the sequence (i.e., parsing, extracting information then representing it with a vector)
    * The decoder takes care of generating a sequence according to the understanding of the encoder (i.e., generating a new language, speech, image, etc.)

Encoder-decoders are:
* Great at sequence-to-sequence tasks: many-to-many such as translation or summarization
* Weights are not necessarily shared across the encoder and decoder
* Input distribution is different from output distribution
* Examples: BART, T5

Sequence-to-sequence models are best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation, or generative question answering.

The pretraining of these models can be done using the objectives of encoder or decoder models, but usually involves something a bit more complex. For instance, T5 is pretrained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces.





# Chapter 2. Using transformers

## A. Tokenization

Tokenizers are one of the core components of the NLP pipeline. They serve one purpose: to translate text into data that can be processed by the model. Models can only process numbers, so tokenizers need to convert our text inputs to numerical data. Their goal is to find the most meaningful representation — that is, the one that makes the most sense to the model — and, if possible, the smallest representation.

The first step of a pipeline is therefore to convert the text inputs into numbers that the model can make sense of. 
To do this we use a tokenizer, which will be responsible for:
* Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
* Mapping each token to an integer
* Adding additional inputs that may be useful to the model

One way to tokenize is **word-based tokenization**:
- Each word gets assigned an ID, starting from 0 and going up to the size of the vocabulary. The model uses these IDs to identify each word. 
    This creates problems problems:
    * Pretty large vocabularies
    * Different representations for close words (e.g., 'dog' and 'dogs')
- We also need a custom token to represent words that are not in our vocabulary. This is known as the “unknown” token, often represented as ”[UNK]” or ””. 
    Which helps reduce the size of the vocabulary but is a loss of information

One way to reduce the amount of unknown tokens is to go one level deeper, using a **character-based tokenizer**:
- With English: From 170,000 words to 256 characters
    However: 
    * Huge loss of information as characters alone have little information, depending on the language (meaningless characters in Latin, meaningful characters in Chinese)
    * It increases the size of the context: from few words to many characters

To get the best of both worlds, we can use a third technique that combines the two approaches: **subword tokenization**.
- Some rules:
    * Frequently used words should not be split into smaller subwords
    * Rare words should be decomposed into meaningful subwords
Subword tokenization algorithms can identify start of word tokens. Each algorithm has its own tokenization. e.g., the BERT tokenizer: *tokenization* ->> *token* & *##ization*. process. 
- This approach:
    * Reduces vocabulary sizes
    * Shares information accross different words
    * Gives the possibility to understand prefixes and suffixes
    * Recognizes similar tokens


All this preprocessing needs to be done in exactly the same way as when the model was pretrained.




# Chapter 3. Fine-tuning a pretrained model

## A. Dataset.map() method

Benefits of that method:
* The results of the function are cached, so it won't take any time if we re-execute the code.
* It can apply multiprocessing to go faster than applying the function on each element of the dataset.
* It does not load the whole dataset into memory, saving the results as soon as one element is processed.


## B. Dynamic padding

It's when you pad your inputs when the batch is created, to the maximum length of the sentences inside that batch.
The "dynamic" part comes from the fact that the size of each batch is determined at the time of creation, and all your batches might have different shapes as a result.


## C. Collate function

It puts together all the samples in a batch.
You can pass the collate function as an argument of a DataLoader. We used the DataCollatorWithPadding function, which pads all items in a batch so they have the same length.


## D.  What happens when you instantiate one of the AutoModelForXxx classes with a pretrained language model (such as bert-base-uncased) that corresponds to a different task than the one for which it was trained?

The head of the pretrained model is discarded and a new head suitable for the task is inserted instead.
For example, when we used AutoModelForSequenceClassification with bert-base-uncased, we got warnings when instantiating the model. The pretrained head is not used for the sequence classification task, so it's discarded and a new head is instantiated with random weights.


## C. A full training

A sketch of the training loop consists in:
1. Taking a batch of training data and feed it to the model
2. Using labels, we compute a loss 
Those numbers being not useful on their own, so:
3. We use the loss numbers to compute the gradients of our model weights (i.e., the derivative of the loss with respect to each model weights)
4. Those gradients are used by the optimizer to update a model weights and make them a little bit better
We then repeat the process with a new batch of training data





# Chapter 4: Sharing models and tokenizers

## The Model Hub

The Model Hub makes selecting the appropriate model simple, so that using it in any downstream library can be done in a few lines of code.
Let’s say we’re looking for a French-based model that can perform mask filling.
We select the camembert-base checkpoint to try it out. The identifier camembert-base is all we need to start using it! As you’ve seen in previous chapters, we can instantiate it using the pipeline() function.


## Loading checkpoints

The only thing you need to watch out for is that the chosen checkpoint is suitable for the task it’s going to be used for. For example, here we are loading the camembert-base checkpoint in the **fill-mask pipeline**, which is completely fine. But if we were to load this checkpoint in the **text-classification pipeline**, the results would not make any sense because the head of camembert-base is not suitable for this task! We recommend using the task selector in the Hugging Face Hub interface in order to select the appropriate checkpoints.

You can also instantiate the checkpoint using the model architecture directly, such as with:
`
from transformers import CamembertTokenizer, CamembertForMaskedLM
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
`

However, we recommend using the Auto* classes instead, as these are by design architecture-agnostic. While the previous code sample limits users to checkpoints loadable in the CamemBERT architecture, using the Auto* classes makes switching checkpoints simple:
`
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
`


## Building a model card

The model card is a file which is arguably as important as the model and tokenizer files in a model repository. It is the central definition of the model, ensuring reusability by fellow community members and reproducibility of results, and providing a platform on which other members may build their artifacts.

Documenting the training and evaluation process helps others understand what to expect of a model — and providing sufficient information regarding the data that was used and the preprocessing and postprocessing that were done ensures that the limitations, biases, and contexts in which the model is and is not useful can be identified and understood.

Therefore, creating a model card that clearly defines your model is a very important step. Here, we provide some tips that will help you with this. Creating the model card is done through the README.md file you saw earlier, which is a Markdown file.

The “model card” concept originates from a research direction from Google, first shared in the paper “Model Cards for Model Reporting” by Margaret Mitchell et al. A lot of information contained here is based on that paper, and we recommend you take a look at it to understand why model cards are so important in a world that values reproducibility, reusability, and fairness.

The model card usually starts with a very brief, high-level overview of what the model is for, followed by additional details in the following sections:
* Model description
* Intended uses & limitations
* How to use
* Limitations and bias
* Training data
* Training procedure
* Evaluation results

Let’s take a look at what each of these sections should contain.

1. Model description
The model description provides basic details about the model. This includes the architecture, version, if it was introduced in a paper, if an original implementation is available, the author, and general information about the model. Any copyright should be attributed here. General information about training procedures, parameters, and important disclaimers can also be mentioned in this section.

2. Intended uses & limitations
Here you describe the use cases the model is intended for, including the languages, fields, and domains where it can be applied. This section of the model card can also document areas that are known to be out of scope for the model, or where it is likely to perform suboptimally.

3. How to use
This section should include some examples of how to use the model. This can showcase usage of the pipeline() function, usage of the model and tokenizer classes, and any other code you think might be helpful.

4. Training data
This part should indicate which dataset(s) the model was trained on. A brief description of the dataset(s) is also welcome.

5. Training procedure
In this section you should describe all the relevant aspects of training that are useful from a reproducibility perspective. This includes any preprocessing and postprocessing that were done on the data, as well as details such as the number of epochs the model was trained for, the batch size, the learning rate, and so on.

6. Variable and metrics
Here you should describe the metrics you use for evaluation, and the different factors you are mesuring. Mentioning which metric(s) were used, on which dataset and which dataset split, makes it easy to compare you model’s performance compared to that of other models. These should be informed by the previous sections, such as the intended users and use cases.

7. Evaluation results
Finally, provide an indication of how well the model performs on the evaluation dataset. If the model uses a decision threshold, either provide the decision threshold used in the evaluation, or provide details on evaluation at different thresholds for the intended uses.


Note
Model cards are not a requirement when publishing models, and you don’t need to include all of the sections described above when you make one. However, explicit documentation of the model can only benefit future users, so we recommend that you fill in as many of the sections as possible to the best of your knowledge and ability.

Model card metadata
If you have done a little exploring of the Hugging Face Hub, you should have seen that some models belong to certain categories: you can filter them by tasks, languages, libraries, and more. The categories a model belongs to are identified according to the metadata you add in the model card header.
This metadata is parsed by the Hugging Face Hub, which then identifies this model as being a French model, with an MIT license, trained on the Oscar dataset.

The full model card specification allows specifying languages, licenses, tags, datasets, metrics, as well as the evaluation results the model obtained when training.


## The push_to_hub() method

* All tokenizers have the push_to_hub method, and using it will push all the tokenizer files (vocabulary, architecture of the tokenizer, etc.) to a given repo.
* All model configurations have the push_to_hub method, and using it will push them to a given repo.
* All models have the push_to_hub method, and using it will push them and their configuration files to a given repo.
* The Trainer also implements the push_to_hub method, and using it will upload the model, its configuration, the tokenizer, and a model card draft to a given repo.





# Chapter 5: The 🤗 Datasets Library

## Filter and transform datasets
🤗 Datasets provides several built-in methods to filter and transform a dataset:
* Shuffle and spit
* Select and filter
* Rename, remove and flatten
* Map


## The load_dataset() function in 🤗 Datasets

It allows you to load a dataset from locally (e.g;, on the laptop), from the Hugging Face Hub or from a remote server


## To produce a random sample from dataset

`from datasets import load_dataset`
`dataset = load_dataset("glue", "mrpc", split="train")`
`dataset.shuffle().select(range(50))`

You first shuffle the dataset and then select the samples from it.


## Memory mapping

It is a mapping between RAM and filesystem storage.
🤗 Datasets treats each dataset as a memory-mapped file. This allows the library to access and operate on elements of the dataset without needing to fully load it into memory.

Some of the main benefits of memory mapping are:
*  Accessing memory-mapped files is faster than reading from or writing to disk.
* Applications can access segments of data in an extremely large file without having to read the whole file into RAM first.


## Text embeddings

Text embeddings represents text as an array of vectors called 'vector'. To create these, we usually use encoder models such as *BERT*.

To compare sentences, we use a similarity metric between two embedding vectors. These vectors usually live in high-dimensional spaces so similarity metrics can be anything that measure distance between vectors.
One very popular metric is **cosine similarity**: it uses the angle between two vectors to measure how close they are. A smaller angle means more similar

One problem is that transformer models like *BERT* will return one vector per token. But we want a single embedding vector for each sentence. To deal with that, we can use a technique called **pooling**. Multiple pooling methods exist, such as mean pooling that averages across all tokens. With mean pooling, we just have to make sure we exclude padding tokens

Once we have our sentence embeddings, we can compute the cosine similarity for each pair of vectors.


## using embeddings for semantic search

Transformer-based language models represent each token in a span of text as an embedding vector. It turns out that one can “pool” the individual embeddings to create a vector representation for whole sentences, paragraphs, or (in some cases) documents. These embeddings can then be used to find similar documents in the corpus by computing the dot-product similarity (or some other similarity metric) between each embedding and returning the documents with the greatest overlap.


## Semantic search

Semantic search is a way to search for matching documents by understanding the contextual meaning of a query.
It uses embedding vectors to represent queries and documents, and uses a similarity metric to measure the amount of overlap between them. 

It is also a way to improve search accuracy.
Semantic search engines can capture the intent of a query much better than keyword matching and typically retrieve documents with higher precision.

Asymmetric semantic seach usually has a short query and a longer paragraph that answers the query.





# Chapter 6: The 🤗 Tokenizer Library

When training a model from scratch.

When performing tokenization, we lose some information. For instance, the size of spaces may not be taken into account; sometimes it's not obviousto know chich word a token belogs to, depending on the tokenizers we use.

e.g., sentence: ````'tokenizers superpowers'````
BERT tokenizer: ```['token', '##izer', '##s', 'super', '##power', '##s']```
RoBERTa Tokenizer: ```['Ġtoken', 'izers', 'Ġ', 'Ġ', 'Ġ', 'Ġ', 'Ġ', 'Ġsuper', 'powers'```

## Tokenizer

A tokenizer will not be suitable if it has been trained on a corpus that is **not similar** to the one you will use to train your model from scratch.
You may want to consider training a new tokenizer so that you have a tokenizer suitable for the training corpus used to train a language model from scratch.

Indeed, dissimilarities can arise from:
- New language
- New characters
- New domain
- New style
- New vocabulary (e.g., medicine vocabulary such as *phrayngitis* or *paracetamol*; python code; etc.)

Unknow (not recognized) tokens are problematic because we cannot recover any information from them.

### Procedure 

The procedure for training a tokenizer can be summarized in main steps:
1. Gathering a corpus of texts
2. Choose a tokenizer architecture
3. Train the tokenizer on the corpus
4. Save the results

You could choose an architecture similar to that of similar models already trained or you can build a totally new tokenizer.
The transformers library provides an easy to use method to train a tokenizer using a know architecture on a new corpus. This way we avoid starting entirely from scratch. The only change is the vocabulary, which will be determined by the training on our corpus.

## The AutoTokenizer method

AutoTokenizer.train_new_from_iterator() only works if the tokenizer you are using is a “fast” tokenizer. 
The 🤗 Transformers library contains two types of tokenizers: some are written purely in Python and others (the fast ones) are backed by the 🤗 Tokenizers library, which is written in the Rust programming language. 
Python is the language most often used for data science and deep learning applications, but when anything needs to be parallelized to be fast, it has to be written in another language. For instance, the matrix multiplications that are at the core of the model computation are written in CUDA, an optimized C library for GPUs.

Training a brand new tokenizer in pure Python would be excruciatingly slow, which is why was developed the 🤗 Tokenizers library. 
Note that just as you don't have to learn the CUDA language to be able to execute your model on a batch of inputs on a GPU, you don't need to learn Rust to use a fast tokenizer. 
The 🤗 Tokenizers library provides Python bindings for many methods that internally call some piece of code in Rust; for example, to parallelize the training of your new tokenizer or, as we saw in Chapter 3, the tokenization of a batch of inputs.

Most of the Transformer models have a fast tokenizer available (there are some exceptions that you can check here: https://huggingface.co/docs/transformers/index#supported-frameworks), and the AutoTokenizer API always selects the fast tokenizer for you if it’s available.

## Fast and Slow tokenizers

Slow tokenizers are those written in Python inside the 🤗 Transformers library, while the fast versions are the ones provided by 🤗 Tokenizers, which are written in Rust.
⚠️ When tokenizing a single sentence, you won’t always see a difference in speed between the slow and fast versions of the same tokenizer. In fact, the fast version might actually be slower! It’s only when tokenizing lots of texts in parallel at the same time that you will be able to clearly see the difference.

The output of a tokenizer isn’t a simple Python dictionary; what we get is actually a special BatchEncoding object. It’s a subclass of a dictionary (which is why we were able to index into that result without any problem before), but with additional methods that are mostly used by fast tokenizers.

Besides their parallelization capabilities, the key functionality of fast tokenizers is that they always keep track of the original span of texts the final tokens come from — a feature we call offset mapping. This in turn unlocks features like mapping each word to the tokens it generated or mapping each character of the original text to the token it’s inside, and vice versa.

Therefore, properly using a fast tokenizer requires giving it multiple texts at the same time. Therefore employing batching=True.

To enable this, fast tokenizers store additional information at each step in the internal pipeline:
- Normalization
- Pre-tokenization
- Model
- Special tokens
They keep track of each span of text.

## The question-answering pipeline

The question-answering pipeline finds the answer to questions in a given context. It also works for very long contexts.

It follows the general steps of the pipeline we say before:
- Tokenizer
- Model
- Postprocessing

Contraty to other tasks, to answer questions the model from the pipeline outputs two tensors: start_logits and end_logits.

To get how it works: we pass tokens representing the question and the context. The context contains some words, the information we are looking for, and more words.
We ask the model to predict which tokens starts the answer and which ends it.
The start_logits vector defines the beginning of the response we are looking for, whereas the end_logits vector defines the end of our response.

To convert the logits to probabilities, we apply the SoftMax. 
But before that, we mask the tokens/logits that are outside of the context, i.e., our question.

An answer is a pair of start and end positions.
After finding the possible answer with the best score, we use the offset mappings to find the corresponding answer in the context.

When the context is very long, it might get truncated by the tokenizer. So, we create several features for different pieces of the context.
Each feature contains the question and a chunk of text from the context.
We might end up with answers split into two or more features. To avoid truncating the answer, we allow some overlap between the features, to make sure at least one of the chunks fully contains the answer to the question.

The tokenizer automatically does it. The **stride** argument controls the number of overlapping tokens.

We then select the most likely answer for each feature and the final answer is the one with the best score.

We could take the argmax of the start and end probabilities — but we might end up with a start index that is greater than the end index, so we need to take a few more precautions. We will compute the probabilities of each possible start_index and end_index where start_index <= end_index, then take the tuple (start_index, end_index) with the highest probability.

Assuming the events “The answer starts at start_index” and “The answer ends at end_index” to be independent, the probability that the answer starts at start_index and ends at end_index is:
start_probabilities[start_index] * end_probabilities[end_index]

So, to compute all the scores, we just need to compute all the products start_probabilities[start_index] * end_probabilities[end_index] where start_index <= end_index.


## Normalization and pre-tokenization

Before splitting a text into subtokens (according to its model), the tokenizer performs two steps: normalization and pre-tokenization.

### Normalization

When a sentence is diverse in its use of characters, e.g., with many special characters, the normalization process reduces the diversity - each tokenizer its normalized set of characters. Therefore the normalization process of a given tokenizer will output characters differently than another tokenizer.

Fast tokenizers provide easy access to the normalization operation.
The normalization operation is automatically included when you tokenize a text.

Some normalizations may not be visible to the eye but can change many things for the computer: different successions of 0 and 1 can render the same character, however the computer works with 0 and 1.
There are some Unicode normalization standards however (e.g., NFC, NFD, NFKC, NFKD) that can erase some of this difference. These standards are often used by tokenizers.

But beware as not all normalizations are suitable for all corpus.
e.g., "un père indigné" can become "un père indigne" after normalization.

### Pre-tokenization

It comes after the normalization of the text and before the application of the tokenization algorithms.
It consists in applying rules that don't need to be learned to perform a division of the text.

The pre-tokenization applies rules to realize a first split of the text.
Each tokenizer its pre-tokenization process.

Pre-tokenization can modify text - such as replacing a space with a special underscore - and split text into tokens.


## Byte-Pair encoding tokenization

Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pretraining the GPT model. It’s used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa.

1. BPE training starts by computing the unique set of words used in the corpus (after the normalization and pre-tokenization steps are completed), then building the vocabulary by taking all the symbols used to write those words.
As a very simple example, let’s say our corpus uses these five words: 
"hug", "pug", "pun", "bun", "hugs"

The base vocabulary will then be ["b", "g", "h", "n", "p", "s", "u"]. For real-world cases, that base vocabulary will contain all the ASCII characters, at the very least, and probably some Unicode characters as well. 
If an example you are tokenizing uses a character that is not in the training corpus, that character will be converted to the unknown token. That’s one reason why lots of NLP models are very bad at analyzing content with emojis, for instance.

2. After getting this base vocabulary, we add new tokens until the desired vocabulary size is reached by learning merges, which are rules to merge two elements of the existing vocabulary together into a new one. 
So, at the beginning these merges will create tokens with two characters, and then, as training progresses, longer subwords.
At any step during the tokenizer training, the BPE algorithm will search for the most frequent pair of existing tokens (by “pair,” here we mean two consecutive tokens in a word). That most frequent pair is the one that will be merged, and we rinse and repeat for the next step.

Going back to our previous example, let’s assume the words had the following frequencies: 
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

Meaning "hug" was present 10 times in the corpus, "pug" 5 times, "pun" 12 times, "bun" 4 times, and "hugs" 5 times. 
We start the training by splitting each word into characters (the ones that form our initial vocabulary) so we can see each word as a list of tokens:
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)

Then we look at pairs. The pair ("h", "u") is present in the words "hug" and "hugs", so 15 times total in the corpus. It’s not the most frequent pair, though: that honor belongs to ("u", "g"), which is present in "hug", "pug", and "hugs", for a grand total of 20 times in the vocabulary.

Thus, the first merge rule learned by the tokenizer is ("u", "g") -> "ug", which means that "ug" will be added to the vocabulary, and the pair should be merged in all the words of the corpus. At the end of this stage, the vocabulary and corpus look like this:
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)

Now we have some pairs that result in a token longer than two characters: the pair ("h", "ug"), for instance (present 15 times in the corpus). The most frequent pair at this stage is ("u", "n"), however, present 16 times in the corpus, so the second merge rule learned is ("u", "n") -> "un". Adding that to the vocabulary and merging all existing occurrences leads us to:
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)

Now the most frequent pair is ("h", "ug"), so we learn the merge rule ("h", "ug") -> "hug", which gives us our first three-letter token. After the merge, the corpus looks like this:
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]
Corpus: ("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)

And we continue like this until we reach the desired vocabulary size.

### Tokenization algorithm

Tokenization follows the training process closely, in the sense that new inputs are tokenized by applying the following steps:
- Normalization
- Pre-tokenization
- Splitting the words into individual characters
- Applying the merge rules learned in order on those splits


## WordPiece tokenization

WordPiece is the tokenization algorithm Google developed to pretrain BERT. It has since been reused in quite a few Transformer models based on BERT, such as DistilBERT, MobileBERT, Funnel Transformers, and MPNET. It’s very similar to BPE in terms of the training, but the actual tokenization is done differently.

⚠️ Google never open-sourced its implementation of the training algorithm of WordPiece, so what follows is our best guess based on the published literature. It may not be 100% accurate

The learning strategy for a WordPiece tokenizer is similar to that of BPE but differs in the way the score for each candidate token is calculated.
To tokenize a text with a learned WordPiece tokenizer, we look for the longest token present at the beginning of the text.

Like BPE, WordPiece starts from a small vocabulary including the special tokens used by the model and the initial alphabet. Since it identifies subwords by adding a prefix (like ## for BERT), each word is initially split by adding that prefix to all the characters inside the word. So, for instance, "word" gets split like this:
w ##o ##r ##d

Thus, the initial alphabet contains all the characters present at the beginning of a word and the characters present inside a word preceded by the WordPiece prefix.

Then, again like BPE, WordPiece learns merge rules. The main difference is the way the pair to be merged is selected. Instead of selecting the most frequent pair, WordPiece computes a score for each pair, using the following formula:
score = (freq_or_pair)/(freq_of_first_element * freq_of_second_element)

By dividing the frequency of the pair by the product of the frequencies of each of its parts, the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary. For instance, it won’t necessarily merge ("un", "##able") even if that pair occurs very frequently in the vocabulary, because the two pairs "un" and "##able" will likely each appear in a lot of other words and have a high frequency. In contrast, a pair like ("hu", "##gging") will probably be merged faster (assuming the word “hugging” appears often in the vocabulary) since "hu" and "##gging" are likely to be less frequent individually.

### Tokenization algorithm

Tokenization differs in WordPiece and BPE in that WordPiece only saves the final vocabulary, not the merge rules learned. Starting from the word to tokenize, WordPiece finds the longest subword that is in the vocabulary, then splits on it. For instance, if we use the vocabulary learned in the example above, for the word "hugs" the longest subword starting from the beginning that is inside the vocabulary is "hug", so we split there and get ["hug", "##s"]. We then continue with "##s", which is in the vocabulary, so the tokenization of "hugs" is ["hug", "##s"].

With BPE, we would have applied the merges learned in order and tokenized this as ["hu", "##gs"], so the encoding is different.

As another example, let’s see how the word "bugs" would be tokenized. "b" is the longest subword starting at the beginning of the word that is in the vocabulary, so we split there and get ["b", "##ugs"]. Then "##u" is the longest subword starting at the beginning of "##ugs" that is in the vocabulary, so we split there and get ["b", "##u, "##gs"]. Finally, "##gs" is in the vocabulary, so this last list is the tokenization of "bugs".

When the tokenization gets to a stage where it’s not possible to find a subword in the vocabulary, the whole word is tokenized as unknown — so, for instance, "mug" would be tokenized as ["[UNK]"], as would "bum" (even if we can begin with "b" and "##u", "##m" is not the vocabulary, and the resulting tokenization will just be ["[UNK]"], not ["b", "##u", "[UNK]"]). This is another difference from BPE, which would only classify the individual characters not in the vocabulary as unknown.


## Unigram tokenization

The Unigram algorithm is often used in SentencePiece, which is the tokenization algorithm used by models like AlBERT, T5, mBART, Big Bird, and XLNet.

The overall training strategy is to start with a very large vocabulary and then iteratively reduce it. At each iteration we calculate a loss on our training corpus using the model. We can use it to choose how to reduce our vocabulary. We will choose to remove what increases the loss the less.


Unigram model is a type of Statistical Language Model (SLM) assuming that the occurence of each word is independent of its previous word.
A SLM will assign a probability to a text considering that the text is in fact a sequence of tokens.
The simplest sequence of tokens to imagine are the words that compose a sentence or the characters.

The particularity of Unigram LM is that it assumes that the occurence of each word is independent of its previous word.
This "assumption" allows us to write that the probability of a text is equal to the product of the probabilities of the tokens that compose it.

It is a very simple model that is not adapted to the generation of text though, since it would always generate the same token, the one which has the greatest probability.
It is however very handy for tokenization.


In more details:
Compared to BPE and WordPiece, Unigram works in the other direction: it starts from a big vocabulary and removes tokens from it until it reaches the desired vocabulary size. There are several options to use to build that base vocabulary: we can take the most common substrings in pre-tokenized words, for instance, or apply BPE on the initial corpus with a large vocabulary size.

At each step of the training, the Unigram algorithm computes a loss over the corpus given the current vocabulary. Then, for each symbol in the vocabulary, the algorithm computes how much the overall loss would increase if the symbol was removed, and looks for the symbols that would increase it the least. Those symbols have a lower effect on the overall loss over the corpus, so in a sense they are “less needed” and are the best candidates for removal.

This is all a very costly operation, so we don’t just remove the single symbol associated with the lowest loss increase, but the p(p being a hyperparameter you can control, usually 10 or 20) percent of the symbols associated with the lowest loss increase. This process is then repeated until the vocabulary has reached the desired size.

Note that we never remove the base characters, to make sure any word can be tokenized.

Now, this is still a bit vague: the main part of the algorithm is to compute a loss over the corpus and see how it changes when we remove some tokens from the vocabulary, but we haven’t explained how to do this yet. This step relies on the tokenization algorithm of a Unigram model, so we’ll dive into this next.

### Tokenization algorithm

A Unigram model is a type of language model that considers each token to be independent of the tokens before it. It’s the simplest language model, in the sense that the probability of token X given the previous context is just the probability of token X. So, if we used a Unigram language model to generate text, we would always predict the most common token.

The probability of a given token is its frequency (the number of times we find it) in the original corpus, divided by the sum of all frequencies of all tokens in the vocabulary (to make sure the probabilities sum up to 1). For instance, "ug" is present in "hug", "pug", and "hugs", so it has a frequency of 20 in our corpus.







# Chapter 7: Main NLP Tasks

## Preprocessing

## Masked language modeling

For both auto-regressive and masked language modeling, a common preprocessing step is to concatenate all the examples and then split the whole corpus into chunks of equal size. This is quite different from our usual approach, where we simply tokenize individual examples. 
Why concatenate everything together? The reason is that individual examples might get truncated if they’re too long, and that would result in losing information that might be useful for the language modeling task.

Instead, we could generate several blocks of context length from that very long text. This can be done by asking the tokenizer to return the overflowing tokens.
Ans if the texts have various lengths, one option is to concatenate them all, then taking chunks of context length.

## Labeling

### Language modeling

Unlike other tasks like text classification or question answering where we’re given a labeled corpus to train on, with language modeling we don’t have any explicit labels. So how do we determine what makes a good language model? Like with the autocorrect feature in your phone, a good language model is one that assigns high probabilities to sentences that are grammatically correct, and low probabilities to nonsense sentences. To give you a better idea of what this looks like, you can find whole sets of “autocorrect fails” online, where the model in a person’s phone has produced some rather funny (and often inappropriate) completions.

Assuming our test set consists mostly of sentences that are grammatically correct, then one way to measure the quality of our language model is to calculate the probabilities it assigns to the next word in all the sentences of the test set. High probabilities indicates that the model is not “surprised” or “perplexed” by the unseen examples, and suggests it has learned the basic patterns of grammar in the language. 


## Metrics

### Cross-entropy, Perplexity and loss functions

Perplexity (PPL) is a common metric used to evaluate language models. The smaller its value, the better the performance.
In machine learning, the likelihood is a common quantity. We can calculate the likelihood of a **sequence** as the product of each element's (i.e., token) probaiblity.
Meaning that for each token, we use the Language Models to predict its probability based on the previous tokens. In the end, we multiply all probabilities to get the likelihood.

Cross-entropy (log-loss function) is closely related to likelihood. It is often used as a loss function in classification. 
In language modeling we predict the next token based on the previous token, which is also a classification task. Therefore to calculate the cross-entropy of a example, we can simply pass it to the model with the inputs as labels. The loss then corresponds to the cross entropy.

Now, to calculate the perplexity, we only need to exponentiate the cross-entropy. The perplexity is therefore closely related to the loss.
This is equivalent to exponentiating the negative average log probability of each token.

Keep in mind that the loss is only a weak proxy for a model's ability to generate quality text, and the same is true for perplexity.
One usually also calculates more sophisticated metrics such as BLEU or ROUGE on generic tasks.

Also, mind that there are various mathematical definitions of perplexity.


## Domain adaptation

When fine-tuning a pretrained model on a new dataset, the fine-tuned model we obtain will make predicitons that are attuned to this new dataset.
When the two models (pretrained model & finetuned model) are trained with the same task, we can compare their prediction on the same input.
The predictions of the two models that will different in a way that reflects the differences between the two datasets. A phenomenon called **domain adaptation**.

Usually, a pretrained model makes generic predictions, whereas the fine-tuned model has its predictions linked what it was fine-tuned on.
However, even if the fine-tuned model adapts to the new dataset, it's not forgetting what it was pretrained on.
