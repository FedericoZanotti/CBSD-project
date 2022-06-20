# Classification and Reconstruction of dishonest of the IADQ questionnaire

**Source Code**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedericoZanotti/CBSD-project/blob/master/CBSD_Project_IADQ.ipynb)

**Little Demo**:  [![CBSD-project](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/federicozanotti/cbsd-project/main/app.py)

This was a very difficult project for the course *Cognitive Behavioral and Social Data*. The professor had just given some datasets to the student and no more information. In my opinion the work my colleagues and me made was over too much, in the sense that we tried everything and in particular we implemented a different approach w.r.t the other students. 

The project has **two purposes**: the first one was to recognize if a subject had lied answering a psycological questionnaire. This is very interesting, because there are no way to identify a lier, except for some control questions that psycologists insert in the questionnaires. The second purpose was to reconstruct the honest answers for a dishonest subject. This was really challenging and to be honest our work did not resolve this problem, but I think we faced in an original way (expecially with the method *Brute Force* we implemented). 
Although the project was really interesting the dataset is too little to create an algorithm that generalizes well. Infact how can we state if a subject lies in a dataset of 225 questionnaire answers? In my opinion, this was also a little bit unethical but the aims were good.

In this project we analized the answers of the [IADQ questionnaire](https://www.traumameasuresglobal.com/_files/ugd/be25b4_31238f0f34b7496aa15e804cc5279a01.pdf) and the participants compiled the test honestly and dishonestly. For the dishonest compiling they had to exaggerate their answers in order to be classified with IADQ disorder. The dataset is composed by 225 participants, so 450 answers with dishonest and honest counterpart. **Python** is used as a programming language

The description of the project with relative source code is well written in the file *CBSD_Project_IADQ.ipynb* and is composed by:
- Statistical analysis
- Cluster of liers
- honest-dishonest classification
- Reconstruction (with many models employed!)

We also builted a simple Demo, using *Streamlit*. In order to fully test it a well-formatted file *test.cvs* is provided.
The main idea of this app is to create a web application that a future psychologist can use in order to see if the patiens are dishonest and what would be their true answers.








