# DKT

A simple visualization of DKT， using flask + vue + pytorch.


## Model

A simple DKT model implemented only by using RNN. 
DKT.pth is based on **assist2009_updated** dataset, which includes 110 kinds of problems(knowledges)  


## About visualization

we only use 6 questions among 110 to make a radar chart, which are:
- 7：Mode
- 8：Mean
- 15：Fraction of

- 92：Rotations
- 59：Exponents
- 50：Pythagorean Theorem

Among them 7、8、15 are the similar knowledges，and the rest of three questions are from different knowledges.

The choices and question index of question form are fixed 

-------
