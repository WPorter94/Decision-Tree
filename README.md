# Decision-Tree


Description:

This Python project implements a decision tree classifier from scratch, designed to handle categorical target variables. The decision tree is constructed using the ID3 algorithm, which 
selects the best attribute to split the data based on information gain and entropy calculations. The tree is then used to classify unseen instances.



Files:

decision_tree.py: Contains the main implementation of the decision tree classifier.

mass_towns_2022.csv: Sample CSV file containing the dataset for testing the decision tree.

README.md: This file, providing an overview of the project.


Dependencies:

Python 3.x


Usage:

Ensure you have Python installed on your system.

Download or clone the repository to your local machine.

Navigate to the project directory in the terminal or command prompt.

Run the following command to execute the decision tree classifier on the provided dataset:

python decision_tree.py

The program will output the accuracy of the model on the test set and display the decision tree in ASCII art format.


Implementation Details:

The decision_tree.py file contains classes for DecisionNode, LeafNode, and DecisionTree, representing the nodes of the decision tree and the decision tree itself.

The read_data() function reads the training data from a CSV file and prepares it for processing.

The train_test_split() function randomly splits the dataset into training and test sets.

The decision tree is built using the DecisionTree class, which uses the ID3 algorithm to recursively split the data based on the best attribute.

The test_model() function evaluates the trained model on the test set and calculates accuracy.

The confusion2x2() function generates a normalized confusion matrix for the two classes in the test set.



Note:

This implementation is specifically designed for categorical target variables.

Feel free to modify the min_examples variable in the decision_tree.py file to adjust the minimum number of examples required for a leaf node.

