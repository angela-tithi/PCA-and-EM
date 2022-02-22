# PCA-and-EM
This repository contains a python project that implements Principal Component Analysis (PCA) and Expectation Maximization (EM). It is used to reduce the dimensions of the data and to cluster the data (i..e. to maximize the probabilities of the distributions (posteriors), the samples had been generated from).

The dataset used here, contains 100 dimensions, I have reduced it to two dimensions (the two principal components), where we can find the greatest variance of the dataset. I used the EM (Expectation Maximization) algorithm later, to cluster the data. It takes about 70-80 iterations on average to converge.

To run this project please download the project, it includes -

```console
main.py
data.txt
and the Pipfiles
```

and execute the following commands -

```bash
pipenv install
pipenv shell
python main.py
```

It may take time to converge the EM algorithm, so please hold your patience. After it converges, you can see a window which shows the animation of the clustering flow. I have attached the animation file that I have got from previous execution. It is in **Previous Execution** folder.

This project was actually an assignment I had to submit in the final year at BUET, I have remade the project, you can find the details about this assignment in the **ML2.pdf** file included. I have followed the steps and the formulas mentioned in this assignment specification. Please let me know if you have questions.
