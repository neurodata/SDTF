## Add streaming option to decision tree classifier
#### Describe the workflow you want to enable
The current `DecisionTreeClassifier` is designed for batch learning and thus not capable of handling streaming data. I hope to make the tree builders able to use Hoeffding bound with a specified confidence for node splitting and make streaming classification possible.[[1]](https://doi.org/10.1007/978-1-4612-0865-5_26)

#### Describe your proposed solution
The Very Fast Decision Tree learner (VFDT) or Hoeffding tree algorithm uses the difference between the best split function and the second-best split function at each frontier node. When the minimal number of instances reach the frontier, the program would compute the Hoeffding bound for the criterion used, such as Gini index, information gain or others. If the difference exceeds the Hoeffding bound, the node will become internal and its two children the new frontiers.[[2]](https://doi.org/10.1145/347090.347107)

There is already a [repository](https://github.com/huawei-noah/streamDM) that implements Hoeffding tree with Spark.

Later algorithms like Adaptive Random Forest (ARF) and Streaming Random Forest (SRF) integrated the Hoeffding tree with bagging approaches and demonstrated its forest-level potentials.[[3]](https://doi.org/10.1007/s10994-017-5642-8)[[4]](https://doi.org/10.1109/IDEAS.2007.4318108)

Potential changes needed on:
- [`sklearn/tree/_splitter.pyx`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_splitter.pyx)
- [`sklearn/tree/_criterion.pyx`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pyx)
- [`sklearn/tree/_tree.pyx`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx)
- [`sklearn/tree/_classes.py`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_classes.py)

#### Describe alternatives you've considered, if relevant
The Streaming Parallel Decision Tree (SPDT) algorithm takes another approach of handling streaming data, saving the summary statistics of them and training the decision tree classifier only when necessary.[[5]](https://dl.acm.org/doi/10.5555/1756006.1756034)

There is already a [repository](https://github.com/soundcloud/spdt) that implements SPDT with Spark.

Potential changes needed on:
- [`sklearn/tree/_classes.py`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_classes.py)

#### References
[1] Hoeffding W. (1994) Probability Inequalities for sums of Bounded Random Variables. In: Fisher N.I., Sen P.K. (eds) The Collected Works of Wassily Hoeffding. Springer Series in Statistics (Perspectives in Statistics). Springer, New York, NY. https://doi.org/10.1007/978-1-4612-0865-5_26

[2] Pedro Domingos and Geoff Hulten. 2000. Mining high-speed data streams. In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '00). Association for Computing Machinery, New York, NY, USA, 71–80. https://doi.org/10.1145/347090.347107

[3] Gomes, H.M., Bifet, A., Read, J. et al. Adaptive random forests for evolving data stream classification. Mach Learn 106, 1469–1495 (2017). https://doi.org/10.1007/s10994-017-5642-8

[4] H. Abdulsalam, D. B. Skillicorn and P. Martin, "Streaming Random Forests," 11th International Database Engineering and Applications Symposium (IDEAS 2007), Banff, Alta., 2007, pp. 225-232. https://doi.org/10.1109/IDEAS.2007.4318108

[5] Yael Ben-Haim and Elad Tom-Tov. 2010. A Streaming Parallel Decision Tree Algorithm. J. Mach. Learn. Res. 11 (3/1/2010), 849–872. https://dl.acm.org/doi/10.5555/1756006.1756034
