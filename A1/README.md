# Usage

Since the solution had to be OS-independent i had created two ways of running and testing the submission in case there are issues with python notebooks, i will go through a small tutorial for both.

## 1. Python Notebook (recommended)

1. Inside the VSCode editor, open the A1.ipynb file and press the "Run All" button and all code cells should run and the markdown cells should render.
2. The Notebook is also in PDF version (A1.pdf) but this file is read-only.

## 2. Raw File (CLI)

If theres problems with the notebook there is also a raw python file that does the same thing as the notebook but without the text/report.

1. 
```cli
cd path-to-A1-folder

Python3 A1.py
```

## dependencies troubleshooting

If for some reason theres an error indicating that there are errors even trying to pip install them, this is what i recommend.

```cli
cd path-to-A1-folder

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

Python3 A1.py (or run the notebook)
```