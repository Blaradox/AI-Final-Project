<!--- 
Make sure to fill in the following information before submitting your
assignment. Your grade may be affected if you leave it blank!
File name: proposal.md
Author username(s): sanuk, sloaneat
Date: December 1, 2016
Submission name: Final Project Proposal 
-->
[//]: # (This is a comment)

## AI-Final-Project

Final project for CS 402 Artificial Intelligence at Whitman College.

We will be using support vector machines (SVMs) to analyze the KDD Cup 2010 dataset, hence extending our understanding of decision trees and learning as discussed in the textbook.

SVMs are supervised learning models and non-binary probabilistic binary linear classifiers. 
Essentially, we will be using this technique for predictive modeling by separating the data into training and testing sets.

The KDD cup is an educational data mining competition, hosted by the PSLC DataShop. 
This is quite a prestitigious competetion and there are awards sponsored by Facebook, Elsevier and IBM Research.
We thought that the topics covered by the event were very interesting and that since this was a competition for learning it would be well suited to a school project. 
The competition covers a host of topics, the current competition is especially interesting as it talks about predicting students success in mathematical problems, based upon a host of variables (collected from the students interactions with Intelligent Tutoring Systems). 



> Preprocessing: KDD Cup 2010 is an educational data mining competition. The data comes from Carnegie Learning and DataShop. This is the training set of the first problem: algebra_2008_2009. We have provide a transformed version used by the winner (National Taiwan Univ). Because labels of the competition's testing set are not available, the training data is split into two sets for training and validation. The validation set is called the testing set here. To access the raw data set, please check the above "KDD CUP 2010" link. This data set is only to be used for research purposes. Users please acknowledge the data is from Carnegie Learning and DataShop. [HFY10c]
>
> - \# of classes: 2
> - \# of data: 8,407,752 / 510,302 (testing)
>  \# of features: 20,216,830 / 20,216,830 (testing)
> 
> Files:
> 
> - kdda.bz2
> - kdda.t.bz2 (testing)



### References

- https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
- Stamper, J., Niculescu-Mizil, A., Ritter, S., Gordon, G.J., & Koedinger, K.R. (2010). Algebra I 2008-2009. Challenge data set from KDD Cup 2010 Educational Data Mining Challenge. Find it at http://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp
