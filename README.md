word_learning
=============

This code provides a framework for modeling cross-situational word learning.
The core algorithm implements the model of Fazly et al. (2010), which is an incremental and probabilistic word learner.


The code also includes extensions of this model that allow investigation of:

* Individual differences in word learning. (Nematzadeh et al., 2011, 2012a, 2014a)
* The role of memory and attention in word learning. (Nematzadeh et al., 2012b, 2013)
* The acquisition of a semantic network. (Nematzadeh et al., 2014b)


An extension of this model is used to study novel word generalization (Nematzadeh et al., 2015); the code can be found [here](https://github.com/eringrant/novel_word_generalization).


References:

* Fazly, A., Alishahi, A., & Stevenson, S. (2010).  [A probabilistic computational model of cross-situational word learning](http://onlinelibrary.wiley.com/doi/10.1111/j.1551-6709.2010.01104.x/abstract).  *Cognitive Science*, 34(6), 1017-1063.

* Nematzadeh, A., A. Fazly, & Stevenson, S. (2011). [A computational study of late talking in word-meaning acquisition](https://mindmodeling.org/cogsci2011/papers/0141/paper0141.pdf). In *Proceedings of the 33rd Annual Conference of the Cognitive Science Society*.

* Nematzadeh, A., Fazly, A., & Stevenson, S. (2012a). [Interaction of word learning and semantic category formation in late talking](https://mindmodeling.org/cogsci2012/papers/0364/paper0364.pdf). In *Proceedings of the 34th Annual Conference of the Cognitive Science Society*.

* Nematzadeh, A., Fazly, A., & Stevenson, S. (2012b). [A computational model of memory, attention, and word learning](http://www.aclweb.org/anthology/W12-1708). In *Proceedings of the 3rd Workshop on Cognitive Modeling and Computational Linguistics* (pp. 80-89). Association for Computational Linguistics.

* Nematzadeh, A., Fazly, A., & Stevenson, S. (2013). [Desirable difficulty in learning: A computational investigation](http://csjarchive.cogsci.rpi.edu/Proceedings/2013/papers/0210/paper0210.pdf). In *Proceedings of the 35th Annual Conference of the Cognitive Science Society*.

* Nematzadeh, A., Fazly, A., and Stevenson, S. (2014a). [Structural differences in the semantic networks of simulated word learners](https://mindmodeling.org/cogsci2014/papers/191/paper191.pdf). In *Proceedings of the 36th Annual Conference of the Cognitive Science Society*.

* Nematzadeh, A., Fazly, A., and Stevenson, S. (2014b). [A cognitive model of semantic network learning](http://emnlp2014.org/papers/pdf/EMNLP2014031.pdf). In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*.

* Nematzadeh, A., Grant, E., and Stevenson, S. (2015). [A computational cognitive model of novel word generalization](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP207.pdf). In *Proceedings of the 2015 Conference on Empirical Methods for Natural Language Processing*.


Starter code is provided in `starter/main.py`,
and development and test data are located at `data/input_wn_fu_cs_scaled_categ.dev` and  `data/input_wn_fu_cs_scaled_categ.tst`, respectively.
The gold standard lexicon, which was used to generate the dev/test data, and which can be used to compute metrics such as the acquisition score, is located at `data/all_catf_norm_prob_lexicon_cs.all`.


Requirements: `Python 2`, `numpy`, `scipy`
