# Sentiment Analysis of Amazon Reviews

## Overview:
A review text usually contains emotional information, which is very useful for evaluating the quality of a product. In this Project, we will train a model to classify a sentence to 3 sentiment: positive, negative and neutral. 



## Dataset:
We will use the Amazon Customer Reviews Dataset, which is provided from Amazon. This dataset consists of many classes, and we will use [book review data]( http://snap.stanford.edu/data/amazon/ ) from them, which is about 4.4GB in size.This is a [link](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) including all the information of those data. There are many attributes in this data set, however we will use mainly the review text as training data.

```
amazon_reviews(
product/productId int,
product/title string, 
product/price double,
review/userId string,
review/profileName string,
review/helpfulness int, 
review/score float,
review/time: int,
review/summary string,
review/text string)
```

Here is an example:
```
product/productId: 0826414346
product/title: Dr. Seuss: American Icon
product/price: unknown
review/userId: A30TK6U7DNS82R
review/profileName: Kevin Killian
review/helpfulness: 10/10
review/score: 5.0
review/time: 1095724800
review/summary: Really Enjoyed It
review/text: I don't care much for Dr. Seuss but after reading Philip Nel's book I changed my mind--that's a good testimonial to the power of Rel's writing and thinking. Rel plays Dr. Seuss the ultimate compliment of treating him as a serious poet as well as one of the 20th century's most interesting visual artists, and after reading his book I decided that a trip to the Mandeville Collections of the library at University of California in San Diego was in order, so I could visit some of the incredible Seuss/Geisel holdings they have there.There's almost too much to take in, for, like William Butler Yeats, Seuss led a career that constantly shifted and metamoprhized itself to meet new historical and political cirsumstances, so he seems to have been both a leftist and a conservative at different junctures of his career, both in politics and in art. As Nel shows us, he was once a cartoonist for the fabled PM magazine and, like Andy Warhol, he served his time slaving in the ad business too. All was in the service of amusing and broadening the minds of US children. Nel doesn't hesitate to administer a sound spanking to the Seuss industry that, since his death, has seen fit to license all kinds of awful products including the recent CAT IN THE HAT film with Mike Myers. Oh, what a cat-astrophe!The book is great and I can especially recommend the work of the picture editor who has given us a bounty of good illustrations.
```



## Method:
We will use [CNN]( https://www.aclweb.org/anthology/D14-1181 ) and [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) as main model for classification, which is proved as a powerful model to solve sentiment analysis problem. And we use [Glove](https://nlp.stanford.edu/projects/glove/) and [SSWE](https://www.aclweb.org/anthology/P/P14/P14-1146.xhtml) as word embedding, which is also a important part of NLP problem. In order to compare the result, we will also implement a traditional machine learning model , [Naive Bayes model](https://en.wikipedia.org/wiki/Naive_Bayes_classifier), and evaluate it. 

## Plan:
We divided the whole project into 4 part: data process, training and evaluating Naive Bayes model, training and evaluating CNN model, training and evaluating RNN model.

| Name | Work |
|:----:|:------:|
|Xi  | data process, CNN model |
|Martin | Naive Bayes model |
|Ziyuan | RNN model|

## Details
### part1. data process:
reference: [How to Clean Text for Machine Learning with Python](https://machinelearningmastery.com/clean-text-machine-learning-python/?spm=5176.100239.blogcont225721.32.cAebmf)


In this part, we choose the `review/text` from data set as the text we use to classify; and use `review/score` as label following below 
condition: 
```
if score >= 4.0 label the text as "positive"
if score == 3.0 label the text as "neutral"
if score <= 2.0 label the text as "negative"
```

In order to use a text to do a classification, the first step is to clear up the text, and it consists of 4 main part:
```
1. read text;
2. spilt the text paragrap to sentence;
3. tokenize the sentence to words;
4. extend the abbreviations (eg. "isn't -> is not");
5. convert all words to lower-case, remove stop words and punctuation;
6. write the cleared text to file;
```

Besides, according to different classifier(NB or CNN), there is also small difference:
in CNN, we should keep the original form of words, because different forms such as "-ing" or "-s" have different means, and with word embedding these information can be used in CNN model. But for Naive Bayes model, we can't use these information, so usually we will stem them (eg. liking -> like, words -> word ).

Here is an example :

original sentence:
```
["This is only for Julie Strain fans. It's a collection of her photos -- about 80 pages worth with a nice section of paintings by Olivia.If you're looking for heavy literary content, this isn't the place to find it -- there's only about 2 pages with text and everything else is photos.Bottom line: if you only want one book, the Six Foot One ... is probably a better choice, however, if you like Julie like I like Julie, you won't go wrong on this one either."]
```
cleared-up sentence ( for CNN ):
```
[[['julie', 'strain', 'fans'], ['collection', 'photos', 'pages', 'worth', 'nice', 'section', 'paintings', 'looking', 'heavy', 'literary', 'content', 'place', 'find', 'pages', 'text', 'everything', 'else', 'line', 'want', 'one', 'book', 'six', 'foot', 'one', 'probably', 'better', 'choice', 'however', 'like', 'julie', 'like', 'like', 'julie', 'go', 'wrong', 'one', 'either']]]
```
cleared-up sentence ( for Naive Bayes ):
```
[['julie', 'strain', 'fan', 'collection', 'photo', 'page', 'worth', 'nice', 'section', 'painting', 'look', 'heavy', 'literary', 'content', 'place', 'find', 'page', 'text', 'everything', 'else', 'line', 'want', 'one', 'book', 'six', 'foot', 'one', 'probably', 'good', 'choice', 'however', 'like', 'julie', 'like', 'like', 'julie', 'go', 'wrong', 'one', 'either']]
```

Second step is to map those words to a indexes: we make a dictionary, including every word which appears in text and assign a index for every word. We also created 2 dictionaries, one of them, i.e. id2word, map indexes to words, another,i.e. word2id, map words to indexes. We can use them later on building classifiers. The structure and description of result after processing can be shown as following:

CNN:

| file name | type | description |
|:----:|:----:|:------:|
|CNN_id2word.txt | dict() | dictionary, use id to find word |
|CNN_word2id.txt | dict() | dictionary, use word to find id |
|CNN_sentence_id_list.txt | list() | 3d list, every word in sentence convert to index  |
|CNN_sentence_list.txt | list() | 3d list, see example "cleared-up sentence for CNN" above|

Naive Bayes:

| file name | type | description |
|:----:|:----:|:------:|
|NB_id2word.txt | dict() | dictionary, use id to find word |
|NB_word2id.txt | dict() | dictionary, use word to find id |
|NB_sentence_id_list.txt | list() | 2d list, every word in sentence convert to index  |
|NB_sentence_list.txt | list() | 2d list, see example "cleared-up sentence for Naive Bayes" above|

All the data can be downloaded by link:( [CNN](https://drive.google.com/file/d/1ZqeeNDQ_-_6eMyT-7huQl_z7fqp6hU0g/view?usp=sharing), [NB](https://drive.google.com/file/d/1JwtWwinm_byUzw34gazKgnlOeYAFnXr-/view?usp=sharing) ), and they can be loaded by following python code:
``` python
import json
def ReadListAndDictFromFile(readFileName):
    with open(readFileName, "r",encoding='UTF-8') as f:
        readedList =  json.loads(f.read())
        return readedList
        
#for example
ReadListAndDictFromFile("CNN_sentence_id_list.txt")
```

Then we will encode those word list to numerical type. based on different classification algorithm, the final text will look a little different. For Naive Bayes, using index as encoder is enough. For CNN and RNN, we need a more complex method called "word embedding" to encode. In this project, for CNN model, we will use [Glove](https://pan.baidu.com/s/1qX9uVTE) and [SSWE](https://pan.baidu.com/s/1jIoOFRK) as word embedding. Besides, in order to make model adapt the training set, we also use [gensim](https://radimrehurek.com/gensim/) to train word embedding based on the data we collected. For RNN model, a random initialized word embedding model will be used.

Here is a example shows how word embedding work: 

``` python
# calculate similarity of word vectors
'amazing'
[('incredible', 0.8330461978912354), ('wonderful', 0.6981839537620544), ('awesome', 0.6934306621551514), ('fantastic', 0.6817658543586731), ('remarkable', 0.6695932149887085), ('phenomenal', 0.6470452547073364), ('astounding', 0.6289204359054565), ('marvelous', 0.6220988035202026), ('astonishing', 0.5953347682952881), ('extraordinary', 0.5950510501861572)]

'disgusting'
[('sickening', 0.7087589502334595), ('revolting', 0.6704294085502625), ('vile', 0.6409405469894409), ('repulsive', 0.6092551946640015), ('lewd', 0.5895463824272156), ('repugnant', 0.5865837335586548), ('horrid', 0.5639320015907288), ('obscene', 0.5621051788330078), ('vulgar', 0.559421718120575), ('depraved', 0.5587021708488464)]
```

### part2. CNN for text classification:

[CNN](https://www.aclweb.org/anthology/D14-1181) model is the most powerful neural network nowadays. In this project, we use it as classifier of text sentiment classification task.

From the lecture we have learned how can we use CNN as image classifier. And in order to transfer this idea into text classifier, we firstly map word to a numeric vector, that is so-called word-embedding technique. According to [Kim et al., 2014](https://www.aclweb.org/anthology/D14-1181), we can construct a sentence to a matrix. Then CNN model can be used as usual. Following image shows the architecture of this model.

![](https://github.com/chrisHuxi/ML-project1-Sentiment-Analysis/blob/master/readme_image/CNN.PNG)

Besides, comparing with the CNN model in computer vision, we consider to use different word embedding as different channel as input.
After adjusting parameter, we got a model as shown below:

| Layer | Parameter | Explanation |
|:----:|:------:|:------:|
|Input  | 64 * 3 | 64 words/sentence, 3 channel |
|Embedding  | 50 | 50 dimension word embedding |
|Convolution | input filter size = (2,50,3); output filter size = ( 63, 1, 128) |valid, stride = 1|
|Pooling| 2 * 1 | stride = 1 |
|Convolution | input filter size = (4,50,3); output filter size = ( 61, 1, 128) |valid, stride = 1|
|Pooling| 2 * 1 | stride = 1 |
|Convolution | input filter size = (8,50,3); output filter size = ( 57, 1, 128) |valid, stride = 1|
|Pooling| 2 * 1 | stride = 1 |
|Convolution | input filter size = (16,50,3); output filter size = ( 49, 1, 128) |valid, stride = 1|
|Pooling| 2 * 1 | stride = 1 |
|Dense | 128 | hidden layer nodes = 128 |
|softmax | 3 | output |

training and test on balance data set, 283563 examples. The result shows as following:
```
             precision    recall  f1-score   support

   Negative       0.69      0.66      0.68      9973
   Neutral        0.48      0.52      0.50      8319
   Positive       0.67      0.66      0.67     10065

avg / total       0.62      0.62      0.62     28357
```

### part3. Naive Bayes:
Multinomial Naive Bayes classification algorithm is considered as a baseline for sentiment analysis tasks. The basic idea is to find the probabilities of classes assigned to text by using the joint probabilities of words and classes. Although this algorithm is very simple and it assumes that features are independent which is very naive in case of text classification, in practice naive Bayes performs surprisingly well and is competitive with more sophisticated techniques.

Because of the independence assumption, naive Bayes classifiers are highly scalable and can quickly learn to use high dimensional features with limited training data. This is useful for many real world datasets where the amount of data is small in comparison with the number of features for each individual piece of data, such as speech, text, and image data.

Bayes’ theorem is stated mathematically as the following relationship:

<div align=center><img width="350"  src="https://github.com/chrisHuxi/ML-project1-Sentiment-Analysis/blob/master/readme_image/NB1.png"/></div>

If we add the assumption that each feature is conditionally independent of every other feature can change this formula to:

<div align=center><img width="370"  src="https://github.com/chrisHuxi/ML-project1-Sentiment-Analysis/blob/master/readme_image/NB2.png"/></div>

Since the denominator is constant and we can remove it and just look for maximum score in the classes

<div align=center><img width="400"  src="https://github.com/chrisHuxi/ML-project1-Sentiment-Analysis/blob/master/readme_image/NB3.png"/></div>

To avoid an errors caused by rounding, logarithm of probabilities is usually used:

<div align=center><img width="350"  src="https://github.com/chrisHuxi/ML-project1-Sentiment-Analysis/blob/master/readme_image/NB4.png"/></div>

for Sentiment analysis of book reviews on Amazon using Naive Bayes:
The dataset that we have collected is strongly unbalanced, this caused a lot of problems to get at least decent accuracy for neutral reviews which were just 8.3% of dataset. When we tried to use Multinomial Naive Bayes on the dataset of size 100 000 samples we got the following results: 

```
             precision    recall  f1-score  

   Negative       0.71      0.00      0.00     
   Neutral        0.00      0.00      0.00      
   Positive       0.78      1.00      0.88     

avg / total       0.50      0.33      0.29    
```
Based on the [paper](http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) which recommends following ideas to improve the model:
#### 1. Use bigger dataset
- By using a dataset with 500 000 samples we were finally able to successfully detect some 20% neutral reviews. 
#### 2. Use combination of Multinomial NB and Complement NB 
- The paper suggests to use a combinations of Multinomial NB and Complement NB. The resulting classifier is called one-vs-all-but-one MNB. This classifier is not implemented in the scikit learn library so we modified the scikit implementation of the Multinomial NB to try this algorithm. This helped to get f1-score for negative to 59%, for neutral to 25% and positive to 91%
 
#### 3. Use TF IDF instead of word count
- Paper also suggests to use term frequency instead of counts of the words. This way it will downscale weights for  words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus. 
 
- This assumption however does not work for our dataset and makes our predictions much worse (f1-score for negative to 7%, for neutral to 0% and positive to 89%). 
#### 4. Consider negation of words
- We also tried to transform words like “not” that can negate the following words “not good” -> “not_good”. This way every time “not” occured in the text it was connected to nearest following noun, adjective or verb. This approach also did not make the results better.
#### 5. Undersample dominating classes in training dataset
- In order to boost the f1-score of the class “neutral reviews” we have also decided to make training data balanced (same number of all classes in training data). This approach discutable because we should not modify training data but in our case it helped to get overall better results.

The best result of the NB classifier that we achieved was by using the one-vs-all-but-one MNB on the whole dataset (1 000 001 samples) with the parameters set to:
```
alpha=0.2, 
min_df=10, 
ngram=(1,2) 
Balanced training dataset with the same number of samples for each clases
```
final result is shown below:
```
             precision    recall  f1-score  

   Negative       0.66      0.75      0.70     
   Neutral        0.62      0.51      0.56      
   Positive       0.71      0.74      0.73     

avg / total       0.66      0.66      0.64    
```

### part4. RNN:

Input is BATCH_SIZE(64) padded to MAX_SEQUENCE_LENGTH(100) sentences, each contain exactly MAX_SEQUENCE_LENGTH(100) id of words between 0 and VOCAB_SIZE(10000)

Embedding Layer is learned during training.

RNN is a sequence of neural network blocks that are linked to each others like a chain. Each one is passing a message to a successor.

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

The structure of an LSTM network remains the same as an RNN, whereas the repeating module does more operations. Enhancing the repeating module enables the LSTM network to remember long-term dependencies.

Bidirectional RNN can learn representations from future time steps to better understand the context and eliminate ambiguity.

![](https://cdn-images-1.medium.com/max/1600/1*6QnPUSv_t9BY9Fv8_aLb-Q.png)

Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. It is a very efficient way of performing model averaging with neural networks.

![](https://saurabhmathur96.github.io/deep-learning-notes/images/dropout-neural-net.png)

The model is shown below:

|Layer| Parameter | Explanation |
|:----:|:------:|:------:|
|Embedding| 100 | Embbeding length |
|LSTM | 200 | Bidirectional, 100 hidden units in lstm |
|Dropout | 0.5 | Dorpout rate |
|Dense1| 128 | Output size | 
|Dense2| 16| Output size |
|softmax| 3 | Classes |

Training on 88000 balanced data set, validated on 10000 balanced dataset, tested on 2000 balanced dataset
```
             precision    recall  f1-score  

   Negative       0.38      0.47      0.42     
   Neutral        0.27      0.36      0.37      
   Positive       0.58      0.68      0.63     

avg / total       0.41      0.56      0.47    
```

## Summary:

- We use Amazon book review as dataset, including 1,000,000 reviews
- We use CNN, Naive Bayes, RNN as classifier to classify the sentiment of a text sentence.
- According to F1-score, Naive Bayes model is better then CNN and RNN model, in our case.

