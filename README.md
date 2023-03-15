# evidently-dvc

## Create virtual environment

Create virtual environment named `.venv` and install python libraries
```bash
python3 -m venv .venv
source .venv/bin/activate
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Set up Jupyter Notebook
```bash
python -m ipykernel install --user --name=evidently
jupyter contrib nbextension install --user
jupyter nbextension enable toc2/main
```

## Run pipeline

### Option 1: in Jupyter Notebook

- Launch `Jupyter Notebook`:

    ```bash
    jupyter-notebook
    ```
- Open notebook `notebooks/bicycle_demand_monitoring.ipynb`
- Run pipeline


### Option 2: DVC pipelines

- `train`:

    ```bash
    dvc repro pipelines/train/dvc.yaml
    ```

- `reports`:

    ```bash
    dvc repro pipelines/reports/dvc.yaml
    ```

### View reports

Enter directory `reports/`, open required period folder and launch HTML report in a browser.
