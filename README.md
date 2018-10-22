# Sentiment Analysis of Amazon Reviews

### overview:
A review text usually contains emotional information, which is very useful for evaluating the quality of a product. In this Project, we will train a model to classify a sentence to 3 sentiment: positive, negative and neutral. 



### dataset:
We will use the Amazon Customer Reviews Dataset, which is provided from Amazon. This dataset consists of many classes, and we will use [book review data]( http://snap.stanford.edu/data/amazon/ ) from them, which is about 4.4GB in size.This is a [link](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) including all the information of those data. There are many attributes in this data set, however we will use mainly the review text as training data.

```
amazon_reviews_parquet(
  marketplace string, 
  customer_id string, 
  review_id string, 
  product_id string, 
  product_parent string, 
  product_title string, 
  star_rating int, 
  helpful_votes int, 
  total_votes int, 
  vine string, 
  verified_purchase string, 
  review_headline string, 
  review_body string, 
  review_date bigint, 
  year int)
```


### method:
We will use [CNN]( https://www.aclweb.org/anthology/D14-1181 ) as main model for classification, which is proved as a powerful model to solve sentiment analysis problem. And we use [Glove](https://nlp.stanford.edu/projects/glove/) and [SSWE](https://www.aclweb.org/anthology/P/P14/P14-1146.xhtml) as word embedding, which is also a important part of NLP problem. In order to compare the result, we will also implement a Naive Bayes model and evaluate it.

### plan:
We divided the whole project into 3 part: data process, training and evaluating Naive Bayes model, training and evaluating CNN model.

| Name | Work |
|:----:|:------:|
|Xi  | data process |
|Martin | Naive Bayes model |
|Ziyuan | CNN model|

 
 
