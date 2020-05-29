# Machine Learning Scoring for Cross Languages Programming Assignments by AST based and Modified TF-IDF Vectorization.

# Requirements:
- Python 3.6.
- Scikit-learn 0.22.2.
- Keras 2.2.5.

# Instruction for Replication:

We put all of our code, data and result inside replicationPackage folder. This folder has following items:

- data: Including ASTData as parsed information about Html and JavaScript, HWInstruction as the 319 homework instruction, HWSubmissions as all 130 students' submission, RubricAndScore as rubric and some sample score's comments, TextData as ccode data of 130 students.
- code: code for running machine learning experiments.
-result: contains all expected result for 3 research questions.

We set the default of any result you generate at the 'result' folder. However, you can go to the code and change the default paths to your expected locations.

The following code will run for evaluating range of programming assignment scores on our pretrained vectors. If you want to generate the vector by yourselves, please run each files in thhe RQ*_Vectorization.

**RQ1. What is the accuracy of original TF-IDF classification on code data?**

```python evaluateRQ1.py```

- Input: the default is the path 'replicationPackage/data/pretrainedVector/TFIDF4/' (you can go to the code and change paths for other vectorization models)
- Output: the accuracy of each systems in 10 ML Classification algorithms. You can see the best accuracy on each systems along with the details prediction in 'details/' folder.


**RQ2. What is the accuracy of modified TF-IDF classification on code data?**

To run for all systems:

```python: RQ1_MLRunning.py```

If you want to run on a specific system such as Moodle, run: 

```python RQ2_MLRunning.py```

**RQ3. What is the accuracy of modified TF-IDF classification on AST nodes?**

```python RQ3_MLRunning.py```
