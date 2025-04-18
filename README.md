# mushroom-classifier

## Project Overview

This project uses machine learning models to classify mushrooms as either **edible** or **poisonous** based on their physical characteristics. The goal is to provide a system that helps foragers, researchers, and enthusiasts identify mushrooms safely.

### Goals:
- Build a machine learning model to classify mushrooms.
- Compare the performance of several models: **Decision Tree**, **Random Forest**, **Logistic Regression**, and **SVM**.
- Evaluate the models using performance metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

---

## Steps to Set Up the Environment and Run the Code

### 1. Clone the repository:
Start by cloning the repository to your local machine.

```bash
git clone https://github.com/drippedtt/mushroom-classifier.git
cd mushroom-classifier

2. Set up a virtual environment:
It's recommended to create a virtual environment to keep dependencies isolated. This helps avoid conflicts between packages. Run the following to create it:

python -m venv venv

3. Activate the virtual environment:
On Windows:

.\venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate

Once activated, your terminal prompt should change to show (venv) indicating the environment is active.

4. Install dependencies:
Now that the virtual environment is active, install the necessary libraries:

pip install -r requirements.txt

5. Run the code:
After the environment is set up and the dependencies are installed, run the main Python script:

python mushroom_classifier.py
This will execute the model training and evaluation and print out the results, including accuracy, precision, recall, and F1-score.

Dependencies and Library Versions
This project requires the following libraries:

pandas

scikit-learn

matplotlib

You can install them via:

pip install -r requirements.txt
Assumptions and Notes
The dataset is clean, with no missing values and no need for imputation.

All the features in the dataset are categorical and non-numeric, so one-hot encoding is used to convert them into a format that machine learning models can process.

The project evaluates models for classification based on the key metrics of accuracy, precision, recall, and F1-score.

