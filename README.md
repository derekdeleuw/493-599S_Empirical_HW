# CSE 493S/599S: Empirical Machine Learning Homework
# Team Members: Krishna Deshpande, Derek De Leuw, Bhagyashree Wagh, Vaibhav Paranji, Will Gannon

# Description: This is a README.md containing the project description and instructions on how to run this project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  

# Install required dependencies and packages
pip install -r requirements.txt

# Run parts 0 and 1: Training a Transformer Model to learn Modular Artihmetic and Study Grokking Phenomenon

# Run all training experiments and generate plots
python train.py

# Test a model interactively
python inference.py

# Run parts 2 and 3: AIME Evaluation

# Open Jupyter notebook (requires A100/H100 GPU. Open in Google Colab if needed. Perform cell by cell execution) 
jupyter notebook part_2_starter.ipynb


