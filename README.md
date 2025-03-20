# Privacy preserving calculation of AUC score

This project was part of the practical course Secure Processing of Medical Data: Privacy-Enhancing Technologies in Practice.
The goal was to calculate the AUC score in a privacy preserving way. The imagined scenario is that multiple parties want to compute the AUC score on their combined data without disclosing the data to each other. We chose Fully Homomorphic encryption and Differential Privacy as two different approaches to accomplish the task.

# Requirements
In order to run the code you need 
- matplotlib
- numpy
- [Pyfhel](https://github.com/ibarrond/Pyfhel)
- scikit-learn

Install with: ``` pip install matplotlib numpy Pyfhel scikit-learn ```
