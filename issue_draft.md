## Add streaming option to decision tree
#### Describe the workflow you want to enable
The current `sklearn.tree` module is designed for batch learning and thus not capable of handling streaming data. I hope to make the tree builders capable of updating the existing tree and make streaming classification possible.

#### Describe your proposed solution
The Very Fast Decision Tree learner (VFDT) or Hoeffding tree algorithm uses the difference between the best split function and the second-best split function at each frontier node. When the minimal number of instances reach the frontier, the program would compute the Hoeffding bound for the criterion used, such as Gini index, information gain or others. If the difference exceeds the Hoeffding bound, the node will become internal and its two children the new frontiers.[[1]](https://doi.org/10.1007/978-1-4612-0865-5_26)[[2]](https://doi.org/10.1145/347090.347107)

Later algorithms like Adaptive Random Forest (ARF) and Streaming Random Forest (SRF) integrated the Hoeffding tree with bagging approaches and demonstrated its forest-level potentials.[[3]](https://doi.org/10.1007/s10994-017-5642-8)[[4]](https://doi.org/10.1109/IDEAS.2007.4318108)

##### First step
Implement the update function in `TreeBuilder` for modifying the existing tree with new training samples. This step would assume consistent shapes (features and classes) of training samples.

Potential changes needed on:
- [`sklearn/tree/_tree.pyx`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx)
- [`sklearn/tree/_classes.py`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_classes.py)

##### Second step
Implement the Hoeffding bound in `Splitter` for alternative ways of node splitting.

Potential changes needed on:
- [`sklearn/tree/_splitter.pyx`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_splitter.pyx)
- [`sklearn/tree/_tree.pyx`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx)
- [`sklearn/tree/_classes.py`](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_classes.py)

##### Further step
Implement online bagging methods and potentially forest-level ensembles.

##### Link for my fork & branch: https://github.com/PSSF23/scikit-learn-stream/tree/stream

#### References
[1] Hoeffding W. (1994) Probability Inequalities for sums of Bounded Random Variables. In: Fisher N.I., Sen P.K. (eds) The Collected Works of Wassily Hoeffding. Springer Series in Statistics (Perspectives in Statistics). Springer, New York, NY. https://doi.org/10.1007/978-1-4612-0865-5_26

[2] Pedro Domingos and Geoff Hulten. 2000. Mining high-speed data streams. In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '00). Association for Computing Machinery, New York, NY, USA, 71–80. https://doi.org/10.1145/347090.347107

[3] Gomes, H.M., Bifet, A., Read, J. et al. Adaptive random forests for evolving data stream classification. Mach Learn 106, 1469–1495 (2017). https://doi.org/10.1007/s10994-017-5642-8

[4] H. Abdulsalam, D. B. Skillicorn and P. Martin, "Streaming Random Forests," 11th International Database Engineering and Applications Symposium (IDEAS 2007), Banff, Alta., 2007, pp. 225-232. https://doi.org/10.1109/IDEAS.2007.4318108
