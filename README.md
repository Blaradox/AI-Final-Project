## AI-Final-Project

#### Final project for CS 402 Artificial Intelligence at Whitman College. 
#### Authors: Kanupria Sanuk, Austin Sloane
<!--- 
Make sure to fill in the following information before submitting your
assignment. Your grade may be affected if you leave it blank!
File name: proposal.md
Author username(s): sanuk, sloaneat
Date: December 1, 2016
Submission name: Final Project Proposal 
-->
[//]: # (This is a comment)

Files:

- fproject.py
- haberman.csv
- haberman.data
- haberman.names
- readme.pdf
- readme.markdown

We will be using support vector machines (SVMs) to analyze [Haberman's Survival data set](https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival) from the UCI Machine Learning Repository. Haberman's data concerns the outcomes of several breast cancer surgeries performed at the University of Chicago's Billings Hospital. Here is an excerpt from "haberman.names" that concerns the data set: 
 > Relevant Information:
>    The dataset contains cases from a study that was conducted between
   1958 and 1970 at the University of Chicago's Billings Hospital on
   the survival of patients who had undergone surgery for breast
   cancer.
> 
> Number of Instances: 306
> 
> Number of Attributes: 4 (including the class attribute)
> 
> Attribute Information: 
> 
>   1. Age of patient at time of operation (numerical)
>   2. Patient's year of operation (year - 1900, numerical)
>   3. Number of positive axillary nodes detected (numerical)
>   4. Survival status (class attribute)
>         1 = the patient survived 5 years or longer
>         2 = the patient died within 5 year

SVMs are supervised learning models and non-binary probabilistic binary linear classifiers. 
Essentially, we will be using this technique for predictive modeling by separating the data into training and testing sets.

The package we will be using for the SVM is the [sklearn package](http://scikit-learn.org/stable/index.html). This package will allow us to do a host of things with our data set. It will allow us to view our expected and predicted outputs, evaluate the accuracy of our model, and even separate the data into training and testing sets for us. 

To use the sklearn package follow this [guide](http://scikit-learn.org/stable/install.html) or alternatively input the following 3 lines in the terminal:

1.  `sudo apt-get install python3-numpy`
2.  `sudo apt-get install python3-scipy`
3. `sudo apt-get install python3-sklearn`

**Note there are some alternative packages for the modules we used in sklearn and the package we use needs sklearn version 0.17.0-1**

We originally were going to analyze the data from the KDD cup, which is an educational data mining competition, hosted by the PSLC DataShop. We thought that the topics covered by the event were very interesting and that since this was a competition for learning it would be well suited to a school project. 
However, SVM's are used to classify data whose attributes are exclusively float values. When looking through the KDD data set we found there were quite a few string values. For an easy example as to why string values for an attribute cannot easily be changed into floats, say we had these values for an attribute:

- `'science'`
- `'math'`
- `'physics' `

If we were to simpy assign numerical values to each of these attributes then we would have: 

- `'science' = 1`
- `'math' = 2`
- `'physics' = 3 `

where `'science'` is further away from `'physics'` than it is from `'math'`. This makes no logical sense. The three string values are three distinct options that must be chosen between. If we assign float values in the way described above, if the model were to try and predict an answer between `science` and `physics` it wouldn't choose either of those options, instead it would choose `math` because the model interprets `math` as  _between_ those two values. Obviously this same problem does not occur with attributes that have a binary value, as we only have two possibilities `0` and `1`. When looking at the data set, there were plenty of floats and binary attributes, however in the total of 480+ attributes, there were too many string attributes that couldn't be converted to floats, so we had to choose another data set. 

We then stumbled upon Haberman's Survival data set, which was comprised of attributes with exclusively float values, so we decided to use this data. As we were focusing on how to use the sklearn package, we wanted to devote the majority of our time to this, and not converting our csv file into a usable format. 
Happily this allowed us to use the `sklearn.SVC` module again. 
We would have needed to use a multi-class classification for the KDD data set. The aim of the KDD problem was to classify donors and how much they would donate;
to do this we would need brackets for how much someone donated (`<$100, $100 - $200, >$200`) which requires multiple classes. 
The SVC module uses an expensive, but accurate, "one against one" approach, which would be too costly to use with the size of the KDD data set.
This "one against one" approach is so expensive because if `n_classes` is the number of classes, then `n_classes*(n_classes-1)/2` classifiers are created.
Therefore, the more accurate our model became, the longer and longer our program took to complete. 

Now when using the sklearn package there are a lot of things you have to tweak before your model becomes accurate. We first noted that when performing the split of our data into a training set and a testing set the default `test_size=0.25` led to some poor results. Our data set is quite small relative to other problems, and it is quite likely that when selecting a testing set (because of the random nature of `train_test_split`) that we would select data points without any variation. Therefore, we decided to increase the size of our test set, by setting `test_size=0.33`. Again, since our data set was small, we decided to increase our cache size from the default `cache_size=200` to `cache_size=1000`. Another thing to consider is that the underlying LinearSVC implementation uses a random number generator; it does this to select features when fitting the model, and therefore different results can appear for the same imput data. Initiall we were getting very different results each time we ran the program so we lowered our tolerance to a very low `tol=0.00001`. And obviously choosing the correct kernel is very important when constructing your SVM. We tested out different kernels and quickly realized that `kernel='sigmoid'` was too inaccurate and when we set the kernel to be a polynomial (even of degree 2) the time taken to complete the program increased drastically. Choosing between `kernel='linear'` and `kernel='rbf'` (where `rbf` stands for radial basis function) was a bit more difficult. Both were similarly accurate, however upon a review of the graph of the data it seemed that `kernel='linear'` was more appropriate. There was also the possibility to use a [Gram matrix](http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use) to find a better kernel. However, when using this kernel there were no drastic improvements to the performance of our program, so we thought it would be best to use a more common kernel for clarity. Then finally after having chosen a proper kernel, we decided to modify `C=1.0`. The lower the `C` value the _smoother_ the decision surface will be, and the higher a `C` value is the more accurate the test examples will be classified. Our hypothesis was that we would want to use a higher `C` value; there are only two possible classifications for our data set (survived or didn't survive, morbid I know!) therefore it makes sense that we'd want to be as accurate as possible so as not to miss a positive classification. This was correct, however we ran into some interesting results: with `C=0.5` we had an average precision of around 50% accuracy, but as we increased to `C=1.0`, `C=10.0` and `C=100.0` our average precision topped when `C=1.0` at 65%-70% and decreased to around 55%-60% for the other `C` values. 

### References

- Haberman, S. J. (1976). Generalized Residuals for Log-Linear Models, Proceedings of the 9th International Biometrics Conference, Boston, pp. 104-122. 

- Landwehr, J. M., Pregibon, D., and Shoemaker, A. C. (1984), Graphical Models for Assessing Logistic Regression Models (with discussion), Journal of the American Statistical Association 79: 61-83. 
[Web Link](http://rexa.info/paper/883f49956b1f22c2c7a435c7f87704e30245ea55)

- Lo, W.-D. (1993). Logistic Regression Trees, PhD thesis, Department of Statistics, University of Wisconsin, Madison, WI. 
[Web Link](http://rexa.info/paper/4f2ee312e02a9897433db0f1631f74b5f7bf56e6)

- http://scikit-learn.org/stable/modules/svm.html
