In this project, I am building a library to recreate the work of Athey, Tibsirani and Wager, Generalized random forests:
https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Generalized-random-forests/10.1214/18-AOS1709.pdf
with the entry point api being src/main/scala/treebased/api/GeneralizedRandomForest.scala

Also, I have implemented the Causal forests algorithm from the earlier paper from Athey and Wager, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"
with the entry point being at src/main/scala/treebased/api/CausalForest.scala
