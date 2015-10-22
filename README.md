word_learning
=============

This code provides a framework for modeling cross-situational word learning.
The core algorithm implements the model of Fazly et. al (2010), which is an
incremental and probabilistic word learner.


The code also includes the extensions of this model which allows investigation of:

* Individual differences in word learning. Nematzadeh et. el (2011, 2012a, 2014a)
* The role of memory and attention in word learning. Nematzadeh et. al (2012b, 2013)
* Acquisition of a semantic network. Nematzadeh et. al (2014b) 

An extension of this model is used to study novel word generalization (Nematzadeh et. al 2015). You can find the code here:


References:
Fazly, A., Alishahi, A., & Stevenson, S. (2010).  A Probabilistic Computational Model of Cross-Situational Word Learning.  *Cognitive Science*, 34(6), 1017-1063.




Starter code is provided in `starter/main.py`,
and development and test data are located at
`data/input_wn_fu_cs_scaled_categ.dev` and 
`data/input_wn_fu_cs_scaled_categ.tst`, respectively.

Requirements: `Python 2`, `numpy`, `scipy`
